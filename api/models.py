import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database import Base


class Project(Base):
    """Проект — загруженный датасет с конфигурацией колонок."""

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    csv_key: Mapped[str] = mapped_column(String(512), nullable=False)
    panel_col: Mapped[str] = mapped_column(String(100), nullable=False)
    date_col: Mapped[str] = mapped_column(String(100), nullable=False)
    value_col: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="project", order_by="Job.created_at")


class Job(Base):
    """Задача обработки данных для проекта."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped["Project"] = relationship("Project", back_populates="jobs")
