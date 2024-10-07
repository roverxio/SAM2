## Containerisation

### Build image
```bash
docker build -t rendernet/sam2:v1.0.0 .
```

### Run Container
```bash
docker run --gpus all -p 8080:8080 --runtime=nvidia rendernet/sam2:v1.0.0
```