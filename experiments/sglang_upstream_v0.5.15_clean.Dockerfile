ARG BASE_IMAGE=tiancij/sglang-upstream-base:v0.5.15-official-local
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="tiancij SGLang upstream baseline"
LABEL org.opencontainers.image.description="Owned immutable clone of the official SGLang v0.5.15 image for clean upstream performance baselines"
LABEL org.opencontainers.image.created="2026-07-17"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
LABEL org.opencontainers.image.version="v0.5.15-clean"
LABEL experiment.owner="tiancij"
LABEL experiment.purpose="clean-upstream-performance-baseline"
LABEL experiment.base.image_id="sha256:06e0aa8359f56ab7a316b60900e6d0dff9bfbb4190d9ce1f5c8caa27e875ae2f"
LABEL experiment.base.repo_digest="sha256:655d004115e384b56b73ad6869bb04b29cce7ba3e597ce8f92eeeb49e2acb0af"
LABEL experiment.base.sglang_commit="f63458b5beaceabbd9d749b9fc956370e1b649e6"
