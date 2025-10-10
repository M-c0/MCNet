# MCNet: Modality-Aware Token Filtering and Common Feature Enhancement Network

MCNet is a Transformer-based framework for **multi-modal vehicle re-identification (ReID)**, designed to address background interference and modality inconsistency in RGB, NIR, and TIR images.  
It introduces **Attention-based Spatial Token Filtering (ASTF)**, **Common Feature Extraction (CFE)**, and **Modality Enhancement (ME)** modules to achieve robust cross-modal feature learning.

## 📦 Code Release
The full code and training scripts will be **released soon**. 🚀  

---

## 🚀 Features
- **Transformer-based multi-modal ReID** framework.
- **ASTF**: token-level filtering to suppress background noise.  
- **CFE**: learn modality-invariant features with global contrastive and local alignment losses.  
- **ME**: selectively enhance modalities with cross-attention.  
- State-of-the-art performance on **RGBNT100** and **MSVR310** datasets.

---
## 📊 Results

Comparisons with state-of-the-art methods on **RGBNT100** and **MSVR310**.  
Transformer-based methods are marked with an asterisk (*).  
Best results are in **bold**, second-best are _underlined_.

### RGBNT100
| Method        | Reference        | mAP   | Rank-1 | Rank-5 | Rank-10 |
|---------------|------------------|-------|--------|--------|---------|
| OSNet         | ICCV'19          | 75.0  | 95.6   | **97.0** | **97.4** |
| HRCN          | ICCV'21          | 67.1  | 91.8   | 93.1   | 93.8    |
| AGW           | TPAMI'21         | 73.1  | 92.7   | 94.3   | 94.9    |
| TransReID*    | ICCV'21          | 75.6  | 92.9   | 93.9   | 94.6    |
| HAMNet        | AAAI'20          | 74.5  | 93.3   | 94.5   | 95.2    |
| PFNet         | AAAI'21          | 68.1  | 94.1   | 95.3   | 96.0    |
| CCNet         | ArXiv'22         | 77.2  | 96.3   | -      | -       |
| GraFT*        | ArXiv'23         | 76.6  | 94.3   | 95.3   | 96.0    |
| GPFNet        | TITS'23          | 75.0  | 94.5   | 95.5   | 95.9    |
| TOP-ReID*     | AAAI'24          | 81.2  | 96.4   | -      | -       |
| FACENet*      | INFFUS'25        | *81.5* | **96.9** | - | - |
| LRMM          | ESWA'25          | 78.6  | *96.7* | -    | -       |
| **MCNet*** (Ours) | -            | **82.1** | 96.1   | *96.5* | *96.7* |

---

### MSVR310
| Method        | Reference        | mAP   | Rank-1 | Rank-5 | Rank-10 |
|---------------|------------------|-------|--------|--------|---------|
| OSNet         | ICCV'19          | 28.7  | 44.8   | *66.2* | 73.1    |
| HRCN          | ICCV'21          | 23.4  | 44.2   | 66.0   | *73.9* |
| AGW           | TPAMI'21         | 28.9  | 46.9   | 64.3   | 72.3    |
| TransReID*    | ICCV'21          | 18.4  | 29.6   | 62.4   | 70.7    |
| HAMNet        | AAAI'20          | 27.1  | 42.3   | 61.6   | 69.5    |
| PFNet         | AAAI'21          | 23.5  | 37.4   | 57.0   | 67.3    |
| CCNet         | ArXiv'22         | 36.4  | *55.2* | -    | -       |
| TOP-ReID*     | AAAI'24          | 35.9  | 44.6   | -      | -       |
| FACENet*      | INFFUS'25        | 36.2  | 54.1   | -      | -       |
| LRMM          | ESWA'25          | *36.7* | 49.7 | -    | -       |
| **MCNet*** (Ours) | -            | **38.7** | **57.5** | **72.1** | **77.5** |

---
