# Plate-state class taxonomy

**Three classes** (`clean`, `finished`, `full`). The label is defined by the plate's
**visible state** only — never by cutlery or by whether a diner is present. The binary
action is **non-monotonic** by design.

| Idx | Class | Visible state | Action |
|---|---|---|---|
| 0 | `clean` | Pristine, **unused** plate — no food, no crumbs, no residue | **do not clear** |
| 1 | `finished` | Meal is **over**: eaten bare (crumbs / sauce smears / residue) **or** only small leftovers / garbage (napkin, paper) | **clear** |
| 2 | `full` | A **moderate-to-full** serving of food | **do not clear** |

## Binary decision rule
- `clear` = { `finished` }
- `do not clear` = { `clean`, `full` }

**Non-monotonic on purpose.** A `clean` plate has the *least* on it yet maps to
*do not clear* (a freshly set plate — the diner is about to eat), while `finished` (some
food / scraps) maps to *clear*, and `full` (the most food) maps back to *do not clear*. So
"clear" is a band in the *middle* of the food-amount axis, not a simple threshold. This is
the main reason for the fine-grained model: a plain "how full is the plate" binary
classifier cannot represent this boundary.

## From 5 generation classes → 3 training classes
The synthetic data was *generated* with a finer 5-class scheme (it gives the diffusion model
clearer, less ambiguous targets), then **consolidated to 3 classes** for training:

| Generation class (in `utils.py` / `prompts.json`) | → Training class |
|---|---|
| `clean` | `clean` |
| `empty` (used, eaten bare) | `finished` |
| `finished_leftovers` (small leftovers / garbage) | `finished` |
| `full` (moderate-to-full) | `full` |
| `unclassified` (too degraded to read) | **dropped** |

So `finished` merges the old `empty` + `finished_leftovers` (both mean "the meal is done →
clear"), and the `unclassified` abstain class was dropped. `utils.py` still defines the
5-class generation lineage; the **trained model and all of 03–05 use the 3 classes above**,
read directly from the class folders on disk.

## Notes for data generation
- **Cutlery is a nuisance attribute**, varied randomly across all classes so the model keys
  on food amount, not on cutlery presence.
- The **`clean` vs `finished`** boundary is the subtlest one to watch under heavy degradation
  (a pristine plate vs an eaten-bare plate with faint crumbs can blur together).
- The degradation in `03_degrade_and_augment` is **class-agnostic** (it never sees the label),
  so no degradation parameter can correlate with the class and become a shortcut.
