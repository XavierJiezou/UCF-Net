# UCF-Net Project Page

An academic project page for **Harnessing CLIP and DINO: An Uncertainty-Aware Cascaded Fusion Network for Generalizable Deepfake Image Detection**.

## Preview locally

From this directory, run:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Contents

- Paper title, author list, affiliations, and direct resource links (PDF, Code, Models, Datasets)
- Abstract and key benchmark metrics summary (~4M images, in-domain & cross-domain mAUC)
- Motivation figure highlighting complementary failure modes of CLIP and DINO representations
- Method architecture diagram with hierarchical feature modeling, LEA, and UAF modules
- Unified deepfake benchmark taxonomy (FS, FR, EFS, FE) and recent-generator evaluation pipeline
- Interactive in-domain, cross-domain, and cross-generator evaluation charts and rankings
- Training scale impact curves (10K, 1M, 2M) and ablation study summaries
- Interactive qualitative Grad-CAM and t-SNE feature-space viewers with lightbox inspection
- Copyable BibTeX citation and release assets

The project page is published from the repository's `page` branch.
