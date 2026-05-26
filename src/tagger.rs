use std::io::{self};

use crate::attribute::Attribute;
use crate::context::{Context, Flag, Reset, ViterbiState};
use crate::dataset::{self, Instance, Item};
use crate::model::Model;

/// The tagger provides the functionality for predicting label sequences for input sequences using a model
#[derive(Debug, Clone)]
pub struct Tagger<'a> {
    /// CRF model
    model: &'a Model<'a>,
    /// CRF context
    context: Context,
    /// Number of distinct output labels
    num_labels: u32,
}

impl<'a> Tagger<'a> {
    pub(crate) fn new(model: &'a Model<'a>) -> io::Result<Self> {
        let num_labels = model.num_labels();
        let mut context = Context::new(Flag::VITERBI | Flag::MARGINALS, num_labels, 0);
        context.reset(Reset::TRANS);
        let mut tagger = Self {
            model,
            context,
            num_labels,
        };
        tagger.transition_score()?;
        tagger.context.exp_transition();
        Ok(tagger)
    }

    /// Predict the label sequence for the item sequence.
    pub fn tag<T: AsRef<[Attribute]>>(&self, xseq: &[T]) -> io::Result<Vec<&str>> {
        if xseq.is_empty() {
            return Ok(Vec::new());
        }

        // Build instance from input sequence
        let mut instance = Instance::with_capacity(xseq.len());
        for item in xseq {
            let item: Item = item
                .as_ref()
                .iter()
                .filter_map(|x| {
                    self.model
                        .to_attr_id(&x.name)
                        .map(|id| dataset::Attribute::new(id, x.value))
                })
                .collect();
            instance.push(item, 0);
        }

        // Create ViterbiState and compute state scores into it
        let mut vstate = ViterbiState::new(self.num_labels, instance.num_items);
        self.state_score(&instance, &mut vstate)?;

        // Run Viterbi
        let (label_ids, _score) = self.context.viterbi(&mut vstate);

        let mut labels = Vec::with_capacity(label_ids.len());
        for id in label_ids {
            let label = self.model.to_label(id).unwrap();
            labels.push(label);
        }
        Ok(labels)
    }

    /// Viterbi-based confidence estimate (not a probability).
    ///
    /// Computes a per-token score margin between the best label and the second-best label
    /// under the Viterbi score matrix, then maps it through a sigmoid.
    ///
    /// This is a heuristic confidence measure in log-score (energy) space:
    /// - High values indicate strong separation between top labels
    /// - Low values indicate ambiguity between competing labels
    ///
    /// It does NOT represent a CRF marginal probability and is not globally normalized.
    pub fn tag_with_confidence<T: AsRef<[Attribute]>>(
        &self,
        xseq: &[T],
    ) -> Result<Vec<(&str, f64)>, anyhow::Error> {
        if xseq.is_empty() {
            return Ok(Vec::new());
        }

        let mut instance = Instance::with_capacity(xseq.len());

        for item in xseq {
            let item: Item = item
                .as_ref()
                .iter()
                .filter_map(|x| {
                    self.model
                        .to_attr_id(&x.name)
                        .map(|id| dataset::Attribute::new(id, x.value))
                })
                .collect();
            instance.push(item, 0);
        }

        let mut vstate = ViterbiState::new(self.num_labels, instance.num_items);

        self.state_score(&instance, &mut vstate)?;

        let (label_ids, _score) = self.context.viterbi(&mut vstate);

        let mut result = Vec::with_capacity(label_ids.len());
        let l = self.num_labels as usize;

        for (t, label_id) in label_ids.iter().enumerate() {
            let label = self.model.to_label(*label_id).unwrap();

            let alpha_row = &vstate.alpha_score[l * t..l * (t + 1)];

            let best_score = alpha_row[*label_id as usize];

            let mut second_best = f64::NEG_INFINITY;
            for (j, &s) in alpha_row.iter().enumerate() {
                if j != *label_id as usize && s > second_best {
                    second_best = s;
                }
            }

            let margin = best_score - second_best;
            let conf = 1.0 / (1.0 + (-margin).exp());

            result.push((label, conf));
        }

        Ok(result)
    }

    fn transition_score(&mut self) -> io::Result<()> {
        // Compute transition scores between two labels
        let l = self.num_labels as usize;
        for i in 0..l {
            let trans = &mut self.context.trans[l * i..];
            let edge = self.model.label_ref(i as u32)?;
            for r in 0..edge.num_features {
                // Transition feature from #i to #(feature.target)
                let fid = edge.get(r as usize)?;
                let feature = self.model.feature(fid)?;
                let j = feature.target as usize;
                trans[j] = feature.weight;
                // Also update transposed matrix for cache-friendly Viterbi
                self.context.trans_t[l * j + i] = feature.weight;
            }
        }
        Ok(())
    }

    /// Compute state scores into ViterbiState.state
    fn state_score(&self, instance: &Instance, vstate: &mut ViterbiState) -> io::Result<()> {
        let l = self.num_labels as usize;
        // Loop over the items in the sequence
        for t in 0..instance.num_items as usize {
            let item = &instance.items[t];
            let state_slice = &mut vstate.state[l * t..];
            // Loop over the attributes attached to the item
            for attr in item {
                // Access the list of state features associated with the attribute
                let id = attr.id;
                let attr_ref = self.model.attr_ref(id)?;
                // A scale usually represents the attribute frequency in the item
                let value = attr.value;
                // Loop over the state features associated with the attribute
                for r in 0..attr_ref.num_features as usize {
                    let fid = attr_ref.get(r)?;
                    let feature = self.model.feature(fid)?;
                    state_slice[feature.target as usize] += feature.weight * value;
                }
            }
        }
        Ok(())
    }
}
