from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class LicensePlateRecord(Base):
    __tablename__ = 'license_plate_records'
    id = Column(Integer, primary_key=True, autoincrement=True)
    license_plate = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    status = Column(String, nullable=False)
    check_in_time = Column(DateTime, nullable=True)
    check_out_time = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'license_plate': self.license_plate,
            'image_url': self.image_url,
            'status': self.status,
            'check_in_time': self.check_in_time.isoformat() if self.check_in_time else None,
            'check_out_time': self.check_out_time.isoformat() if self.check_out_time else None,
        }

DATABASE_URL = "sqlite:///car_model.db"

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()