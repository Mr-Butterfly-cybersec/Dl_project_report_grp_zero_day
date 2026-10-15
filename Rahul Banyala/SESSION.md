# Session Handoff

Date: 8 October 2026

## Project objective

The group project investigates four CNN-based facial recognition architectures. Part I compares the architectures using results reported in their original research papers. Part II will later implement the selected architecture for real-time classification of the group members using the group's own dataset.

Rahul's selected architecture is LightCNN-9.

## Work completed

- Prepared and verified the LaTeX report source in `rahul.tex`.
- Added the approved abstract.
- Added the introduction in the user's direct, human writing style.
- Added the complete LightCNN-9 architecture section.
- Added the Max-Feature-Map equation as a typed LaTeX equation.
- Added the original LightCNN paper as an IEEE reference.
- Created a publication-style LightCNN-9 architecture diagram using Matplotlib and an LFW grayscale input image.
- Inserted the vector PDF diagram as a full-width, two-column figure with a numbered caption and label.
- Added a preliminary Analysis and Discussion method using LFW and YTF.
- Added a full-width comparison table with the LightCNN-9 row completed and placeholder rows for the other three architectures.
- Saved the common writing structure for teammates in `../Setup/what_to_give.txt`.

## Verified LightCNN-9 facts

- Input: `128 x 128 x 1` grayscale face image.
- Architecture: one initial convolution plus four convolution blocks containing two convolutions each, giving nine convolutional layers in total.
- Each convolution block: `1 x 1 convolution -> MFM -> 3 x 3 convolution -> MFM`.
- Four max-pooling operations.
- Flattened feature count: 8,192.
- Fully connected output before MFM: 512.
- Final embedding: 256 dimensions.
- Classification output: `K` identity logits.
- Paper-reported parameters: approximately 5.556 million.
- Paper-reported computation: approximately 1 GFLOP.
- Paper-reported feature extraction time: 67 ms on one core of an Intel i7-4790.
- Paper-reported LFW accuracy: 98.80%.
- Paper-reported YTF accuracy: 93.40%.

The tensor shapes were checked against `LightCNN/light_cnn.py` using a `1 x 1 x 128 x 128` test tensor.

## Important files

- `rahul.tex`: main report source.
- `rahul.pdf`: most recently compiled report.
- `figures/lightcnn9_publication.pdf`: vector diagram used in the report.
- `figures/lightcnn9_publication.png`: high-resolution raster version.
- `figures/lightcnn9_publication.svg`: editable vector version.
- `figures/lightcnn9_publication.py`: reusable Matplotlib diagram generator.
- `LightCNN/light_cnn.py`: original LightCNN architecture implementation.
- `../Setup/what_to_give.txt`: common structure and checklist for all four architecture sections.
- `C:\Users\bash\Desktop\1511.02683v4.pdf`: original LightCNN research paper.
- `C:\Users\bash\Downloads\Mini Project_DL_1.2.pdf`: assignment instructions.

## Current comparison plan

The four architectures will be compared using the same structure:

1. Overview and design objective.
2. Main architectural feature.
3. Input and architecture flow.
4. Training and evaluation setup.
5. Reported performance and efficiency.
6. Strengths, limitations, and suitability for real-time recognition.

The preliminary comparison table uses LFW and YTF. Accuracy values must only be treated as direct comparisons when the papers use the same dataset, protocol, and metric. The table also records parameters, embedding size, inference time, and tested hardware.

## Remaining report work

- Replace the Architecture 2, 3, and 4 placeholders after teammates provide their sections.
- Replace their placeholder rows in the comparison table.
- Confirm whether YTF is reported by enough of the selected architectures. If not, choose another common second dataset or mark unavailable values clearly.
- Complete the critical Analysis and Discussion after all results are available.
- Write the approximately 100-word conclusion after the final comparison.
- Replace the architecture placeholder names in the abstract and introduction.
- Expand the introduction toward the required approximately 800 words if the final page limit allows it.
- Perform a final similarity check and proofreading pass.
- Compile twice after changing citations, labels, figures, or tables.

## Compilation

Run from the `Rahul Banyala` folder:

```powershell
lualatex .\rahul.tex
lualatex .\rahul.tex
```

Two passes resolve citations, table numbers, and figure references. The document compiles successfully. Remaining `Underfull \\hbox` messages are layout warnings from narrow columns, not fatal errors. MiKTeX may also display an unrelated update reminder.

## Latest visual state

The LightCNN-9 diagram appears at the top of page 3 across both columns. Its vector text is clear, the caption is numbered as Figure 1, and the in-text figure reference resolves after two compilation passes.

