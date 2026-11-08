# HeteroFlash test image: Spark vLLM (glm5next, sm_121) + the EXL3 path from the Max-Q control image (b12x 1.3.0 + vLLM exl3 files)
FROM eugr/spark-vllm-b12x:latest
COPY b12x-1.3.0.tgz vllm-exl3-files.tgz /tmp/
RUN SP=/usr/local/lib/python3.12/dist-packages && rm -rf $SP/b12x && cd $SP && tar xzf /tmp/b12x-1.3.0.tgz && tar xzf /tmp/vllm-exl3-files.tgz && rm -f /tmp/*.tgz
