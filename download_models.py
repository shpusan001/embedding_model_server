from transformers import AutoTokenizer, AutoModel

AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", cache_dir="./local_models/labse")
AutoModel.from_pretrained("sentence-transformers/LaBSE", cache_dir="./local_models/labse")
