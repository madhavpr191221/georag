"""FastAPI application for the local M10 retrieval interface."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from georag.m10_service import M10Service, QueryParseError, parse_natural_language
from georag.query import (
    ParseRequest, PlaceChoice, RetrievalRequest, RetrievalResponse,
    SpectralRetrievalRequest, SpectralRetrievalResponse,
)


@lru_cache(maxsize=1)
def _service() -> M10Service:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    return M10Service()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.georag = _service()
    yield


app = FastAPI(title="GeoRAG local retrieval API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _engine() -> M10Service:
    try:
        return _service()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict:
    engine = _engine()
    return {"status": "ready", "dataset": engine.dataset_summary(), "run_count": len(engine.public_runs()),
            "gazetteer_ready": engine.gazetteer_path.is_file()}


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return _engine().public_runs()


@app.get("/api/tiles")
def list_tiles(q: str = "", limit: int = Query(default=100, ge=1, le=200)) -> list[dict]:
    return _engine().catalog(q, limit)


@app.get("/api/places", response_model=list[PlaceChoice])
def list_places(q: str = Query(min_length=1, max_length=120), level: str | None = None,
                country: str | None = None) -> list[PlaceChoice]:
    try:
        return _engine().find_places(q, level, country=country)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503 if isinstance(error, FileNotFoundError) else 422,
                            detail=str(error)) from error


@app.post("/api/parse")
def parse_query(request: ParseRequest) -> dict:
    try:
        parsed = parse_natural_language(request.query)
        return {"query": parsed}
    except QueryParseError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        # Do not return provider request details that might expose account metadata.
        raise HTTPException(status_code=502, detail=f"Query parsing failed: {type(error).__name__}") from error


@app.post("/api/retrieve", response_model=RetrievalResponse)
def retrieve(request: RetrievalRequest) -> dict:
    try:
        engine = _engine()
        if request.query.example_tile_id:
            return engine.retrieve_with_example(request.query, request.run_id, request.k,
                                                request.query.example_tile_id)
        return engine.retrieve(request.query, request.run_id, request.k)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/spectral/retrieve", response_model=SpectralRetrievalResponse)
def retrieve_spectral(request: SpectralRetrievalRequest) -> dict:
    try:
        return _engine().retrieve_spectral(
            request.query, request.k, request.before_example_tile_id, request.after_example_tile_id,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/tiles/{tile_id}/preview")
def tile_preview(tile_id: str) -> Response:
    try:
        return Response(_engine().preview(tile_id), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/tiles/{tile_id}/indices/{index_name}")
def tile_spectral_index(tile_id: str, index_name: str) -> Response:
    try:
        return Response(_engine().spectral_map(tile_id, index_name), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/tiles/{before_id}/change/{after_id}/{index_name}")
def tile_spectral_change(before_id: str, after_id: str, index_name: str) -> Response:
    try:
        return Response(_engine().spectral_map(before_id, index_name, after_id), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
