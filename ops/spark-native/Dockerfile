# HeteroFlash route-3 image: the Spark-native vLLM (glm5next, b12x) with one patch allowing swiglu_limit on IQ2/Q8_0 experts
FROM eugr/spark-vllm-b12x:latest
COPY patch_iq2_limit.py /tmp/
RUN python3 /tmp/patch_iq2_limit.py && rm -f /tmp/patch_iq2_limit.py
