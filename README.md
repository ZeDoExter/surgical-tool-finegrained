# surgical-tool-finegrained

Fine-grained classification ของเครื่องมือทันตกรรม **14 classes** บนถาดสีเงิน — บาง class ต่างกันแค่ความยาว

**Architecture:** `DINOv2-S/14` + `LoRA r=16` + `length fusion` + `ArcFace`

```
image → DINOv2 → 384-d ─┐
                        ├─ concat → Linear → 384 → ArcFace
mask  → length (px/cm) ─┘
```

## วิธีใช้

```bash
pip install -r requirements.txt

# เทรน (default 504 มาจากการทดลอง)
python train.py --data_dir dataset --epochs 50 --batch_size 32 --img_size 504

# ประเมิน
python evaluate.py --checkpoint outputs/best_model.pt --data_dir dataset

# ทำนายรูปเดียว (+ mask)
python infer.py --checkpoint outputs/best_model.pt --image path.jpg --mask_json path.json --ann_id 0
```

Dataset: export แบบ **COCO Segmentation** จาก Roboflow

```
dataset/
  train/_annotations.coco.json + *.jpg  # 1032 samples
  valid/_annotations.coco.json + *.jpg  # 244 samples
```

`1 annotation = 1 sample` — ถ้า 1 รูปมีหลายชิ้น ให้ใช้ `bbox_margin=0.15` เพื่อ crop รายชิ้น

## Colab

เปิด `DentalInstrument_DINOv2_ArcFace.ipynb` — ไฟล์เดียวจบ (เขียน `%%writefile` ครบทุกโมดูล)

- เซลล์ `3.5` เช็ค `kNN probe` ก่อนเทรน
- เซลล์ `4` เทรน (`504 batch32` บน T4)
- เซลล์ `5.5 / 5.6` เทรนและเทียบ `ablation` 6 สูตร

ดูวิธีใช้ละเอียดใน `ALGORITHM.md`

## Config

```python
from config import TrainConfig
cfg = TrainConfig(
    data_dir="dataset",
    img_size=504,      # จากการทดลองให้ผลดี (ต้องหาร 14 ลงตัว)
    batch_size=32,     # T4 15GB, ถ้าใช้ GTX 1650 4GB ให้ลดเป็น 16
    bbox_margin=0.15,  # จากการทดลองให้ผลดี
    lora_r=16,
    calibration_ratio=None,  # cm/px ถ้ามีไม้บรรทัดอ้างอิง
)
```

แก้ใน `config.py` หรือ override ตอนสร้าง `TrainConfig` ก็ได้

## ผลการทดลอง

**kNN probe (frozen, ไม่ได้เทรน) — ใช้เลือก img_size ก่อนเทรน**

| img_size | kNN probe | หมายเหตุ |
|---|---|---|
| 224 | 0.7131 | baseline เดิม |
| 504 | 0.7582 | ใช้เป็น default |
| 518 | 0.7459 |  |
| 560 | 0.7336 |  |
| 546 | 0.7295 |  |
| 616 | 0.7295 |  |

`bbox_margin` ที่ `504`: `0.0 → 0.3074` / `0.10 → 0.7377` / `0.15 → 0.7582` / `0.20 → 0.7336`

**Training (LoRA r16, 50e, early-stop)**

| Split | Val Acc | Balanced | Needle↔Artery |
|---|---|---|---|
| 1032/244 single (224) | 0.9467 | 0.9445 | 5+3=8 |
| 1032/244 single (504) | 0.9426 | 0.9468 | 3+3=6 |
| 1020/256 k-fold Fold1 (224) | 0.9688 | — | 8/256 errors |

> `504` ให้ `kNN` สูงกว่า `224/616` และ `Needle` ลดจาก `8 → 6` — จึงตั้งเป็น default สำหรับเทรนจริง
> ดู `ablation` 6 สูตร (`CAHM/LGMS/SEF`) ใน `tools/evaluate_ablation.py` หรือเซลล์ `5.6` ใน notebook

## License

MIT
