"""FastAPI rating engine.

Run: `uvicorn fremtpl2_pricing.api.app:app` (artefacts from FREMTPL2_ARTIFACTS, default ./artifacts).
Endpoints: GET /health, GET /model-info, POST /quote, POST /quote/batch.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from fremtpl2_pricing.scoring.engine import RatingEngine

DEFAULT_ARTIFACTS = Path(os.environ.get("FREMTPL2_ARTIFACTS", "artifacts"))
MAX_BATCH = int(os.environ.get("FREMTPL2_MAX_BATCH", "1000"))


class Policy(BaseModel):
    """One policy to rate. Ranges and category levels are checked again against the loaded tariff."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "VehPower": 6,
                "VehAge": 4,
                "DrivAge": 35,
                "BonusMalus": 50,
                "VehBrand": "B12",
                "VehGas": "Regular",
                "Density": 1200,
                "Region": "R24",
                "exposure": 1.0,
            }
        },
    )

    VehPower: Annotated[int, Field(ge=1, description="vehicle power class")]
    VehAge: Annotated[int, Field(ge=0, description="vehicle age in years")]
    DrivAge: Annotated[int, Field(ge=16, description="driver age in years")]
    BonusMalus: Annotated[int, Field(ge=1, description="French bonus-malus coefficient (50 = best)")]
    VehBrand: str
    VehGas: str
    Density: Annotated[float, Field(gt=0, description="inhabitants per km2 of the policyholder's municipality")]
    Region: str
    exposure: Annotated[float, Field(gt=0, le=1, description="contract term in years")] = 1.0
    current_premium: Annotated[
        float | None, Field(gt=0, description="premium for the same term today; enables the change cap")
    ] = None


class FactorEffect(BaseModel):
    factor: str
    level: str
    relativity: float


class Quote(BaseModel):
    model_version: str
    frequency: float = Field(description="expected claims per policy-year (full-year cover)")
    severity: float = Field(description="expected capped cost per claim")
    risk_premium_annual: float
    technical_premium_annual: float
    premium_for_term: float
    capped: bool
    top_factors: list[FactorEffect]


class BatchRequest(BaseModel):
    policies: list[Policy] = Field(min_length=1)


class BatchResponse(BaseModel):
    model_version: str
    quotes: list[Quote]


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    directory = artifacts_dir or DEFAULT_ARTIFACTS

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = RatingEngine.load(directory)
        yield

    app = FastAPI(
        title="freMTPL2 rating engine",
        version="1.0",
        lifespan=lifespan,
        description="GLM frequency x severity tariff for French MTPL (portfolio project).",
    )

    def engine(request: Request) -> RatingEngine:
        return request.app.state.engine  # type: ignore[no-any-return]

    def rate(eng: RatingEngine, policies: list[Policy]) -> list[Quote]:
        df = pd.DataFrame([p.model_dump() for p in policies])
        valid, rejected = eng.validate(df)
        if len(rejected):
            detail = [{"index": int(str(i)), "reason": r} for i, r in rejected["reject_reason"].items()]
            raise HTTPException(status_code=422, detail=detail)
        q = eng.quote(valid)
        tops = eng.top_factors(valid)
        return [
            Quote(
                top_factors=[FactorEffect(**t) for t in tops[n]],
                **{str(k): (bool(v) if k == "capped" else v) for k, v in row.items()},  # type: ignore[arg-type]
            )
            for n, (_, row) in enumerate(q.iterrows())
        ]

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "model_version": engine(request).model_version}

    @app.get("/model-info")
    def model_info(request: Request) -> dict[str, Any]:
        return engine(request).info()

    @app.post("/quote", response_model=Quote)
    def quote(policy: Policy, request: Request) -> Quote:
        return rate(engine(request), [policy])[0]

    @app.post("/quote/batch", response_model=BatchResponse)
    def quote_batch(body: BatchRequest, request: Request) -> BatchResponse:
        if len(body.policies) > MAX_BATCH:
            raise HTTPException(status_code=413, detail=f"at most {MAX_BATCH} policies per request")
        eng = engine(request)
        return BatchResponse(model_version=eng.model_version, quotes=rate(eng, body.policies))

    return app


app = create_app()
