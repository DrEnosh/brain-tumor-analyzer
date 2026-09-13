from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    patients: Mapped[list["Patient"]] = relationship(back_populates="owner")


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    age: Mapped[int]
    sex: Mapped[str] = mapped_column(String(16))
    contact: Mapped[str] = mapped_column(String(120), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    owner: Mapped[User] = relationship(back_populates="patients")
    scans: Mapped[list["Scan"]] = relationship(back_populates="patient", order_by="Scan.created_at.desc()",
                                               cascade="all, delete-orphan")

    @property
    def code(self) -> str:
        return f"PT-{self.id:05d}"


class Scan(Base):
    __tablename__ = "scans"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    image_file: Mapped[str] = mapped_column(String(64))
    heatmap_file: Mapped[str] = mapped_column(String(64))
    predicted_label: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float]
    probabilities_json: Mapped[str] = mapped_column(Text)
    attention_json: Mapped[str] = mapped_column(Text)
    colour_warning: Mapped[bool] = mapped_column(default=False)
    atypical_input: Mapped[bool] = mapped_column(default=False)
    ood_score: Mapped[float] = mapped_column(default=0.0)
    model_version: Mapped[str] = mapped_column(String(80))
    inference_ms: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    patient: Mapped[Patient] = relationship(back_populates="scans")


def make_session_factory(database_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):
            dbapi_connection.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
