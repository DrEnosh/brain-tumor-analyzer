### Qualitative review of the heat-maps

Manual review of the two figures above for model `efficientnet_b0-epoch11` (seeded sample, so the figures are reproducible).

**Where the heat-maps look right**
- Meningiomas with a clearly visible enhancing mass: all three sampled cases put the peak directly on the mass (suprasellar, parietal convexity, and superior convexity lesions).
- Pituitary tumours on axial and sagittal slices: attention sits on the central skull-base / sellar region, where the pituitary gland is.

**Where they are weak or misleading**
- Gliomas: attention is broad and centred on the midline rather than tightly on the lesion (e.g. the axial case with a cystic lesion in the upper-left hemisphere only partly overlaps the hot spot).
- "No tumor" predictions: two of the three sampled heat-maps have hot spots in the image corners, outside the head. The model is partly using image-level cues (framing, background, processing style) instead of anatomy. The data explains why: only about 1% of "no tumor" images are 512 × 512 pixels (254 different sizes), while 85–100% of each tumour class are, so image format alone separates "no tumor" from the rest. This is the most important caveat for real-world use.
- The heat-map comes from a 7 × 7 feature map, so it marks a region, not a tumour outline.

**Errors**
- Most confident errors are pituitary tumour ↔ meningioma confusions for masses near the sella turcica, where the two tumour types genuinely look alike. In some of these the heat-map is on the mass (right place, wrong class); in others it is on unrelated brain tissue.
- A meningioma with a small posterior lesion was called "no tumor" and the heat-map misses the lesion. The same scan appears twice (a contrast-shifted copy in the same duplicate group), so it counts as two test errors.
