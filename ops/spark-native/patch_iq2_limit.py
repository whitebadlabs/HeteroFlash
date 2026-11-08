# Allow GLM-5.3-Flash's swiglu_limit (clamp 10.0) on the ModelOpt IQ2/Q8_0 MoE path: B12xExperts already passes the
# limit into b12x ActivationSpec; only alpha/beta (OAI-style SwiGLU) stay rejected.
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt_iq2_xs.py")
s = p.read_text()
old = """            for value in (
                moe_config.swiglu_limit,
                moe_config.swiglu_alpha,
                moe_config.swiglu_beta,
            )"""
assert old in s, "patch anchor not found"
s = s.replace(old, """            for value in (
                moe_config.swiglu_alpha,
                moe_config.swiglu_beta,
            )""", 1)
p.write_text(s)
print("patched")
