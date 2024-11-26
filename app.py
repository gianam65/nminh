from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from pathlib import Path
import time
from PIL import Image
from io import BytesIO
from src.lp_recognition import E2E 
import cloudinary
from cloudinary.uploader import upload
from entity import LicensePlateRecord, session
import time  
from src.data_utils import data_mapper
from datetime import datetime

cloudinary.config(
    cloud_name="dtxvbyskh",
    api_key="531672898283468",
    api_secret="pwpvcEJMQIzZm1cTmam1ub6Ms4g",
    allowed_formats=["png", "jpg"],
)
app = Flask(__name__)
CORS(app)  

model = None

def load_model():
    global model
    model = E2E()  

load_model()

def save_image_to_external_service(path):
    try:
        response = upload(
            path,
            folder="images"
        )

        image_url = response.get("secure_url")

        return image_url
    except Exception as e:
        print(f"Error uploading image to Cloudinary: {e}")

    return ""

def predict_license_plate(image_path):
    img_path = Path(image_path)
    img = cv2.imread(str(img_path))
    
    start = time.time()
    image, license_plate = model.predict(img)
    end = time.time()

    print('Model processed in %.2f s' % (end - start))
    return image, license_plate

@app.route('/upload', methods=['POST'])
def upload_services():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy ảnh được tải lên'}), 400

    file = request.files['image']
    image_path = './upload_services.jpg'  
    file.save(image_path)

    time.sleep(0.3)

    image_url = save_image_to_external_service(image_path)

    return jsonify({'image': image_url }), 200

@app.route('/process_image', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy ảnh được tải lên'}), 400

    file = request.files['image']
    image_path = './uploaded_image.jpg'  
    file.save(image_path)

    image_url = save_image_to_external_service('./uploaded_image.jpg')

    _, license_plate = predict_license_plate(image_path)

    existing_record = session.query(LicensePlateRecord).filter_by(license_plate=license_plate).first()

    if not existing_record:
        new_record = LicensePlateRecord(
            license_plate=license_plate,
            image_url=image_url,
            status='new',  
            check_in_time=datetime.now(),  
            check_out_time=None  
        )
        session.add(new_record)
        session.commit()

    return jsonify({'image': image_url, 'license_plate': license_plate}), 200

@app.route('/records', methods=['GET'])
def get_records():
    records = session.query(LicensePlateRecord).all()
    return jsonify([record.to_dict() for record in records])

@app.route('/record/<int:id>', methods=['GET'])
def get_record(id):
    record = session.query(LicensePlateRecord).get(id)
    if record:
        return jsonify(record.to_dict())
    else:
        return jsonify({'error': 'Không tìm thấy dữ liệu'}), 404

@app.route('/record', methods=['POST'])
def add_record():
    data = request.json
    new_record = LicensePlateRecord(
        license_plate=data['license_plate'],  
        image_url=data['image_url'],          
        status=data['status'],                 
        check_in_time=data.get('check_in_time'),
        check_out_time=data.get('check_out_time') 
    )
    session.add(new_record)
    session.commit()
    return jsonify(new_record.to_dict()), 201

@app.route('/record/<int:id>', methods=['PUT'])
def update_record(id):
    data = request.json
    record = session.query(LicensePlateRecord).get(id)
    if record:
        record.license_plate = data['license_plate']  
        record.image_url = data['image_url']          
        record.status = data['status']                 
        record.check_in_time = data.get('check_in_time')  
        record.check_out_time = data.get('check_out_time') 
        session.commit()
        return jsonify(record.to_dict())
    else:
        return jsonify({'error': 'Không tìm thấy dữ liệu'}), 404

@app.route('/record/<int:id>', methods=['DELETE'])
def delete_record(id):
    record = session.query(LicensePlateRecord).get(id)
    if record:
        session.delete(record)
        session.commit()
        return jsonify({'message': 'Xoá dữ liệu thành công'})
    else:
        return jsonify({'error': 'Không tìm thấy dữ liệu'}), 404



if __name__ == '__main__':
    app.run(debug=True)

