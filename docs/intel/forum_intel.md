# Forum Discussions & Community Intel: rsna-knee-abnormality-detection

| ID | Title | Votes | Comments | Date |
| :--- | :--- | :--- | :--- | :--- |
| 733343 | Knee Abnormality Detection AI Challenge Overview | 83 | 12 | 2026-08-06 15:54:27.876000 |
| 733965 | Use of Commercially Hosted LLMs | 21 | 13 | 2026-08-09 09:52:50.447000 |
| 733375 | Welcome to Knee Abnormality Detection Challenge! | 17 | 1 | 2026-08-06 17:17:57.863000 |
| 730709 | How to get started + Competition's Official Discord | 6 | 1 | 2026-07-30 03:29:19.177000 |
| 735304 | Best single-model score | 65 | 107 | 2026-08-14 23:38:09.839000 |
| 733932 | "Not addressed" is a label too — what we learned reading 4,407 knee reports with an LLM | 55 | 9 | 2026-08-09 05:54:36.206000 |
| 743416 | Please clarify which external knee MRI datasets are allowed | 45 | 5 | 2026-09-26 01:52:06.163000 |
| 733497 | I got a perfect usecase for this competition. | 42 | 20 | 2026-08-07 05:43:30.002000 |
| 733826 | Possible inconsistencies between MRI reports and provided labels | 28 | 2 | 2026-08-08 13:08:26.562000 |
| 736170 | Sharing my knee MRI | 22 | 6 | 2026-08-19 18:21:17.021000 |
| 733652 | Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels from the reports | 21 | 7 | 2026-08-07 18:25:02.787000 |
| 737696 | RSNA Raptor Weights | 19 | 20 | 2026-08-26 23:40:02.832000 |
| 733517 | 0.932 LB within one day. Tested for DICOM metadata shortcut | 17 | 2 | 2026-08-07 07:12:14.851000 |
| 735767 | What Could the Final Ceiling Be for RSNA 2026? | 17 | 12 | 2026-08-18 00:05:14.912000 |
| 735348 | Does Kaggle Do Anything About Byte-for-Byte Public Notebook Theft? | 15 | 14 | 2026-08-15 08:01:05.982000 |
| 735154 | Scaling the encoder bought us nothing (+0.0011) | 15 | 3 | 2026-08-14 08:05:01.209000 |
| 740610 | Plateau at 0.903: sharing my pipeline and ablations, and looking for advice | 15 | 14 | 2026-09-10 12:48:08.313000 |
| 736268 | Too many noise public notebookS. | 13 | 13 | 2026-08-20 04:31:34.063000 |
| 735826 | What is the real bottleneck now: labels, image pipeline, or model capacity? | 13 | 11 | 2026-08-18 09:38:04.704000 |
| 736678 | Maybe worth checking the Efficiency LB? | 13 | 4 | 2026-08-22 01:27:30.292000 |

## High-Value Technical Discussions

### Topic 746792: Reaching 0.949 with one CoAtNet: labels, native crops, and anatomical mirroring (Danial Zakaria / nartaa)
- **Breakthrough:** Single CoAtNet RMLP-2 model trained on 384x384 full-field with SWA (epochs [14, 11, 12]) scores **0.949 Public LB**.
- **Inference Pipeline:** Evaluates 94 three-slice windows with finding-specific multiple-instance attention pooling. Native 320-pixel center crop (`[32:352, 32:352]`).
- **Anatomical Mirror TTA:** Averaging standard view + anatomical left/right mirror:
  - Sagittal: reverse slice channels (flips slices 0 and 2).
  - Coronal & Axial: horizontal image width flip.
  - Average the two probability vectors equally.
- **Negative Results:**
  - Width-flip TTA across every imaging plane hurt score (0.940 -> 0.939).
  - Reducing window density from 94 to 24 dropped score (0.940 -> 0.934).
  - Shrinking inference crop to 288px dropped score (0.948 -> 0.946).
- **OAI eligibility remains unresolved for named datasets/checkpoints:** In [#743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416), the host says simple registration and acceptance of data-use terms meet the general accessibility test, while institutional research agreements or institution-only access do not. Kaggle staff later said it cannot pre-approve specific datasets or weights ([#747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455)); the OAI-specific geo-access question remained unanswered in [#749145](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145) when checked on 2026-10-21. NIMH describes OAI data as Open Access, so this is an uncertainty, not a finding of disqualification. Our Exp-18 and Exp-15 kernels both attach Nartaa CoAtNet weights trained partly with OAI knees; **Exp-15 is not an OAI-free hedge**. Public weights trained on competition data are award-eligible per [#744056](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056), but that does not settle OAI-trained weights. See the current [final-submission selection note](../final-submission-selection.md); no clean-data fallback has been verified.

### Endgame discussion update (2026-10-21)

- [#747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455): Kaggle staff will not interpret third-party data-use terms or pre-approve individual weights; teams remain responsible for compliance.
- [#743420](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743420): No host answer found to whether incompatible upstream licenses exempt winning weights from the public-weight release requirement.
- [#749570](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749570): A community analysis reports heavy public-score ties and roughly 135 teams per 0.001 near the medal band; treat as a dated community snapshot, not official standings.
- [#743356](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743356): A forum response says both final selections can be considered for both Main and Efficiency prizes; no slot needs to be reserved for Efficiency.
- [#748181](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181): Kaggle staff confirms a one-week lock on publishing new public notebooks before the deadline. This does not provide evidence of a score improvement.

### Topic 748011: Local CV or Public LB: which one can we trust? (colum2131 / datnt114)
- **CV Correlation Discovery:**
  - AUC on the 58 gold studies vs Public LB: **$r = 0.66$** (noisy due to small $N=58$).
  - AUC on ~4,300 studies with extracted labels vs Public LB: **$r = 0.87$** (strong linear correlation).
  - Evaluating 5-fold OOF across all 4,407 studies is significantly more reliable than the 58-study gold subset.
- **Current Top Ceiling:** Competitor `datnt114` reported best private LB at **0.961** with local 5-fold OOF of 0.937.

### Topic 747658: Report vs Ground Truth label discrepancies (Heng Cher Keng)
- **Label Independence:** Clinical reports and ground-truth contest annotations were created independently. In study `1.2.826.0.1.3680043.8.498.12606657226568558340797193167488111973`, the ground-truth label has `Effusion: 1`, while the medical report explicitly states *"No knee effusion is present."*
- **Takeaway:** Pure text-based NLP extraction hits a structural glass ceiling. Direct multi-plane MRI visual feature extractors (like our RadImageNet ASL specialist) are mandatory to detect visual abnormalities omitted or discounted by the original reporting radiologist.

### Topic 747553: DINOv3 cannot detect ACL/MCL tears
- **Transformer Texture Blindness:** Vision transformer patch tokens struggle with fine, sub-centimeter linear structures like ACL/MCL tears and cortical hairline fractures.
- **Convolutional Advantage:** Modern pure convolutions (ConvNeXt, ResNet-50) and hybrid CoAtNets dramatically outperform transformers on ligament and bone micro-architecture.

### Topic 735304: Best single-model score

<p>A five-fold ResNet with a resolution of 224px     . 0.954</p>

<p>What was the score on gold ? Also did you focus on better labeling ?</p>

<p>Are you using resnet 18? </p>

<p>What is the best score you’ve achieved with a single model?</p>
<p>I’m trying to train a strong single model first before moving on to ensembling, and I’m curious how far a single model can go.</p>
<p>I’ve noticed that most of the public solutions use ensembles of 20+ models and achieve around 0.91. For those scoring around 0.92–0.94 on the leaderboard, are you also using ensembles, or is it possible to reach that range with a single model?</p>
<p>For reference, my current best score is 0.887 with DINOv2.</p>
