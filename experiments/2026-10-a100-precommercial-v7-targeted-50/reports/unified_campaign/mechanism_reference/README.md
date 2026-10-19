# Cached reference geometry

CPU-only paired geometry of480 fixed reference images. No model inference.

Models with exact pinned caches: 15. Missing models: CLIP ViT-B/32, CLIP ViT-B/16, CLIP ViT-L/14@336, DINOv2-Base.

CKA/RSA compare identical image IDs/order across models. Within-class residual CKA controls shared class-center structure. Layer CKA uses actual cached block IDs.

Joint PCA/t-SNE use all ten reference classes in one fit per model. These plots contain no attacked images. t-SNE distances do not support quantitative claims. Reference associations alone do not establish why an ensemble transfers.
