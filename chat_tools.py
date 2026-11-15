"""Validated tools exposed to the model; economic data stays behind FRED MCP."""

from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from analytics import parse_date
from fred_service import get_comparison_analysis, get_series_analysis


class DateWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    start: str | None = Field(default=None, description="Inclusive YYYY-MM-DD start date, or null.")
    end: str | None = Field(default=None, description="Inclusive YYYY-MM-DD end date, or null.")
    years: int | None = Field(default=None, gt=0, description="Rolling years, or null. Cannot combine with start/end.")

    @model_validator(mode="after")
    def validate_window(self):
        lower = parse_date(self.start) if self.start is not None else None
        upper = parse_date(self.end) if self.end is not None else None
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("start must be on or before end")
        if self.years is not None and (lower is not None or upper is not None):
            raise ValueError("Use years or explicit dates, not both")
        return self

    def bounds(self):
        return {"start": self.start, "end": self.end, "years": self.years}


class SeriesArguments(DateWindow):
    series_id: Literal["GDP", "CPILFESL"]
    include_history: bool = False


class ComparisonArguments(DateWindow):
    include_history: bool = False


class ChartArguments(DateWindow):
    format: Literal["png", "svg", "pdf"] = "png"


TOOL_MODELS = {
    "analyze_economic_series": SeriesArguments,
    "compare_economic_series": ComparisonArguments,
    "create_comparison_chart": ChartArguments,
}
DESCRIPTIONS = {
    "analyze_economic_series": (
        "Retrieve latest available levels, YoY rates, dated historical summaries and optionally full history "
        "for nominal GDP or seasonally adjusted Core CPI. Null dates/years mean all available history; latest "
        "is latest within the selected range. Set include_history for detailed trends. Dates are observation dates."
    ),
    "compare_economic_series": (
        "Compare nominal GDP growth and Core CPI inflation quarterly; return Pearson correlation, coverage, "
        "summaries, definitions and optionally history. CPI uses the mean of all three monthly index levels "
        "before YoY change. Date bounds select whole quarters; unavailable quarters are excluded from correlation. "
        "Use a requested period, or years=10 if the user has not specified a period."
    ),
    "create_comparison_chart": (
        "Create an actual comparison time-series and scatter chart, saved to an automatically named file in outputs/. "
        "Returns its absolute path and the comparison summary. Call only when a visualization is requested. "
        "Date bounds select whole quarters. Default to years=10 when no period is specified."
    ),
}


def tool_definitions():
    definitions = []
    for name, model in TOOL_MODELS.items():
        schema = model.model_json_schema()
        # OpenAI strict schemas require all fields; optional values use null.
        schema["required"] = list(schema["properties"])
        definitions.append({
            "type": "function", "name": name, "description": DESCRIPTIONS[name],
            "parameters": schema, "strict": True,
        })
    return definitions


class EconomicTools:
    def __init__(self, fred_client, output_dir=None):
        self.fred_client = fred_client
        self.output_dir = (
            Path(output_dir) if output_dir is not None
            else Path(__file__).resolve().parent / "outputs"
        )

    async def execute(self, name, arguments):
        model = TOOL_MODELS.get(name)
        if model is None:
            return {"error": "Unknown tool. Use one of the provided economic tools."}
        try:
            args = model.model_validate(arguments)
        except ValidationError:
            return {"error": "Invalid tool arguments. Use GDP or CPILFESL; valid ISO dates; positive integer years OR explicit dates; and only the declared fields."}
        try:
            if name == "analyze_economic_series":
                report = await get_series_analysis(self.fred_client, args.series_id, **args.bounds())
            else:
                bounds = args.bounds()
                if all(value is None for value in bounds.values()):
                    bounds = {"years": 10}
                report = await get_comparison_analysis(self.fred_client, **bounds)
            if name == "create_comparison_chart":
                from visualization import save_comparison_chart
                destination = self.output_dir / f"comparison-{uuid4().hex}.{args.format}"
                report["chart_path"] = str(save_comparison_chart(report, destination))
            if not getattr(args, "include_history", False):
                report = {key: value for key, value in report.items() if key != "history"}
            return report
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception:
            # Transport/SDK errors may contain request details. Never pass them
            # to the model or terminal; return an actionable, credential-free error.
            return {"error": "The FRED request or chart export failed. Check the connection, authorization, and output-directory permissions, then retry."}
