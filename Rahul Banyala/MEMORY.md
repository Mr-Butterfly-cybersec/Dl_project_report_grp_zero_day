# Working Memory

## User preferences

- Work one section at a time.
- Present report wording in chat first and edit `rahul.tex` only after approval.
- Preserve the user's simple, direct, human writing style while correcting grammar and spelling.
- Avoid wording that sounds unnecessarily formal or machine-generated.
- Use original wording and cite the original research paper to reduce plagiarism risk.
- Do not invent experimental results, metrics, hardware, datasets, or missing paper values.
- Clearly distinguish results reported by a paper from results produced by the group.
- Keep all architecture sections in the same structure so the comparison remains consistent.

## Agreed research method

- Part I is a literature-based critical comparison of four architectures.
- Use figures and numerical results from original papers with proper IEEE citations.
- A shared dataset name alone does not make results directly comparable. Dataset split, preprocessing, protocol, and metric must also match.
- Use LFW as the main common benchmark and provisionally use YTF as the second benchmark.
- Compare accuracy or verification performance together with parameter count, computation, embedding size, inference time, and hardware.
- Write `not reported` or use a dash when an original paper does not provide a value.
- Do not treat verification accuracy as classification accuracy.

## LightCNN-9 decisions

- Use the official `network_9layers` implementation in `LightCNN/light_cnn.py` as the architecture reference.
- Describe the code-level `group` modules as `Conv Block 1-4` for clarity.
- Explain MFM as competition between paired feature maps that reduces `2C` channels to `C` using the element-wise maximum.
- Use the Matplotlib-generated vector PDF diagram in the report.
- Use an LFW image only as the input example; the drawn layer volumes are schematic and are not measured activations.
- Later implementation will use transfer learning by retaining the feature extractor and replacing the final classifier for the group's identities.

## Writing and formatting rules

- Report format: A4, double column, 11-point Cambria, 1.15 line spacing, 2 cm margins.
- Section headings: 12-point bold, left aligned, Roman numbering.
- Abstract requirement: 120-200 words.
- Introduction target: approximately 800 words.
- Conclusion target: approximately 100 words.
- Figures and tables require numbers, captions, and in-text references.
- References use IEEE style.
- Use typed LaTeX equations.
- Prefer vector PDF figures for sharp text.

## Resume point

Wait for the other three architecture names, sections, and original-paper results. Then update their section headings and comparison rows, complete the Analysis and Discussion, and write the final conclusion.
