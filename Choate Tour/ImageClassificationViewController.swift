import UIKit
import CoreML
import Vision
import ImageIO

enum ChoateBuildings {
    case Archbold
    case Sign
    case Carl
    case Colony
    case Lanphier
}

struct ChoateBuildingsDescription {
    let Archbold = "Archbold is home to the Admission Office and Head of School and Associate Head of School’s Offices, while the upper two floors serve as a girls’ dorm. Built in 1928, Archbold originally served as the School’s first infirmary."
    let Sign = "This is the sign for the art buildings!"
    let Carl = "The Carl C. Icahn Center for Science was designed by award-winning architect I. M. Pei. Each floor of the three-story building is dedicated to its own discipline: physics, biology, and chemistry. The Center includes 22 classrooms and laboratories; a large reference/study area; a conservatory; and the 150-seat Getz Auditorium."
    let Colony = "Ann and George Colony Hall, the newest addition to the Paul Mellon Arts Center, opened in fall 2019. It is the second academic facility on campus designed by Robert A.M. Stern Architects. To the left, the lobby opens into a series of music practice rooms and classrooms. To the right is the main auditorium. With more than 1,000 seats, the auditorium serves not only as the venue for weekly all-school meetings, but as a performance hall that can accommodate both large and small audiences. Open stairs in the lobby lead both to the auditorium's balcony and to a new dance studio, with changing rooms and an office for the dance program. At the top of the building, a third level entrance leads from the top rows of the balcony to a woodland path along the hillside."
    let Lanphier = "Designed by Pelli Clarke Pelli, this LEED Gold-certified building consists of two classroom wings joined by a glass connector, a corridor that faces one of the School’s two ponds.  The building houses Choate’s i.d.Lab, the Shattuck Robotics Lab, a café, and a collaborative study space. It also connects to the Carl C. Icahn Center for Science by a footbridge over the ponds, bringing together the two buildings."
}

class ImageClassificationViewController: UIViewController {
    // MARK: - IBOutlets
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var cameraButton: UIButton!
    @IBOutlet weak var classificationLabel: UILabel!
    @IBOutlet weak var classificationView: UIVisualEffectView!
    @IBOutlet weak var descriptionView: UIVisualEffectView!
    @IBOutlet weak var descriptionLabel: UILabel!
    
    
    var classificationValues: [ChoateBuildings:Float] = [.Archbold:0, .Carl:0, .Colony:0, .Lanphier:0, .Sign:0]
    
    // MARK: - Overrides
    override func viewDidLoad() {
        descriptionView.isHidden = true
    }
    
    // MARK: - Image Classification
    
    /// - Tag: MLModelSetup
    lazy var classificationRequests: [VNCoreMLRequest] = {
        do {
            let model1 = try VNCoreMLModel(for: ChoateTourBuildingClassifier_1().model)
            let model2 = try VNCoreMLModel(for: ChoateTourBuildingClassifier_2().model)
            let model3 = try VNCoreMLModel(for: ChoateTourBuildingClassifier_3().model)
            let model4 = try VNCoreMLModel(for: ChoateTourBuildingClassifier_4().model)
            let model5 = try VNCoreMLModel(for: ChoateTourBuildingClassifier_5().model)
            let model6 = try VNCoreMLModel(for: ChoateTourBuildingClassifier_6().model)
            let model7 = try VNCoreMLModel(for: ChoateTourBuildingClassifier_7().model)
            
            let request1 = VNCoreMLRequest(model: model1, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            let request2 = VNCoreMLRequest(model: model2, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            let request3 = VNCoreMLRequest(model: model3, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            let request4 = VNCoreMLRequest(model: model4, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            let request5 = VNCoreMLRequest(model: model5, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            let request6 = VNCoreMLRequest(model: model6, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            let request7 = VNCoreMLRequest(model: model7, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            
            request1.imageCropAndScaleOption = .scaleFit
            request2.imageCropAndScaleOption = .scaleFit
            request3.imageCropAndScaleOption = .scaleFit
            request4.imageCropAndScaleOption = .scaleFit
            request5.imageCropAndScaleOption = .scaleFit
            request6.imageCropAndScaleOption = .scaleFit
            request7.imageCropAndScaleOption = .scaleFit
            
            return [request1, request2, request3, request4, request5, request6, request7]
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    func updateClassifications(for image: UIImage) {
        classificationLabel.text = "Classifying..."
        
        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        guard let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                try handler.perform(self.classificationRequests)
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }
    
    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                self.classificationLabel.text = "Unable to classify image.\n\(error!.localizedDescription)"
                return
            }
            
            let classifications = results as! [VNClassificationObservation]
            
            if classifications.isEmpty {
                self.classificationLabel.text = "Nothing recognized."
            } else {
                for classification in classifications {
                    if !classification.confidence.isZero {
                        switch classification.identifier {
                        case "Archbold": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Archbold]! + classification.confidence), forKey: ChoateBuildings.Archbold)
                        case "Archbold-Back": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Archbold]! + classification.confidence), forKey: ChoateBuildings.Archbold)
                        case "Archbold-Close": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Archbold]! + classification.confidence), forKey: ChoateBuildings.Archbold)
                        case "Archbold-Side": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Archbold]! + classification.confidence), forKey: ChoateBuildings.Archbold)
                        case "Art-Center-Sign": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Sign]! + classification.confidence), forKey: ChoateBuildings.Sign)
                        case "Carl-C-Icahn-Center": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Carl]! + classification.confidence), forKey: ChoateBuildings.Carl)
                        case "Carl-C-Icahn-Center-Close": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Carl]! + classification.confidence), forKey: ChoateBuildings.Carl)
                        case "Carl-C-Icahn-Center-Side": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Carl]! + classification.confidence), forKey: ChoateBuildings.Carl)
                        case "Carl-C-Icahn-Center-side": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Carl]! + classification.confidence), forKey: ChoateBuildings.Carl)
                        case "Colony-Hall": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Colony]! + classification.confidence), forKey: ChoateBuildings.Colony)
                        case "Colony-Hall-Back": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Colony]! + classification.confidence), forKey: ChoateBuildings.Colony)
                        case "Colony-Hall-Close": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Colony]! + classification.confidence), forKey: ChoateBuildings.Colony)
                        case "Colony-Hall-Corner": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Colony]! + classification.confidence), forKey: ChoateBuildings.Colony)
                        case "Colony-Hall-Far": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Colony]! + classification.confidence), forKey: ChoateBuildings.Colony)
                        case "Colony-Hall-Side": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Colony]! + classification.confidence), forKey: ChoateBuildings.Colony)
                        case "Colony-Hall-Side-Far": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Colony]! + classification.confidence), forKey: ChoateBuildings.Colony)
                        case "Colony-Hall-Stair": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Colony]! + classification.confidence), forKey: ChoateBuildings.Colony)
                        case "Lanphier-Center": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Lanphier]! + classification.confidence), forKey: ChoateBuildings.Lanphier)
                        case "Lanphier-Center-Side": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Lanphier]! + classification.confidence), forKey: ChoateBuildings.Lanphier)
                        case "Lanphier-Center-Side-2": self.classificationValues.updateValue((self.classificationValues[ChoateBuildings.Lanphier]! + classification.confidence), forKey: ChoateBuildings.Lanphier)
                        default: fatalError()
                        }
                    }
                }
                
                let greatestClassification = self.classificationValues.max { a, b in a.value < b.value }
                
                self.classificationView.isHidden = true
                
                switch greatestClassification!.key {
                case ChoateBuildings.Archbold: self.descriptionLabel.text = "Archbold\n\n\(ChoateBuildingsDescription().Archbold)"
                case ChoateBuildings.Sign: self.descriptionLabel.text = "Sign\n\n\(ChoateBuildingsDescription().Sign)"
                case ChoateBuildings.Carl: self.descriptionLabel.text = "Carl C. Icahn Center for Science\n\n\(ChoateBuildingsDescription().Carl)"
                case ChoateBuildings.Colony: self.descriptionLabel.text = "Ann and George Colony Hall\n\n\(ChoateBuildingsDescription().Colony)"
                case ChoateBuildings.Lanphier: self.descriptionLabel.text = "Lanphier Center for Mathematics and Computer Science\n\n\(ChoateBuildingsDescription().Lanphier)"
                }
                self.descriptionView.isHidden = false
            }
        }
    }
    
    // MARK: - Photo Actions
    
    @IBAction func takePicture() {
        // Show options for the source picker only if the camera is available.
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            presentPhotoPicker(sourceType: .photoLibrary)
            return
        }
        
        let photoSourcePicker = UIAlertController()
        let takePhoto = UIAlertAction(title: "Take Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .camera)
        }
        let choosePhoto = UIAlertAction(title: "Choose Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .photoLibrary)
        }
        
        photoSourcePicker.addAction(takePhoto)
        photoSourcePicker.addAction(choosePhoto)
        photoSourcePicker.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        
        present(photoSourcePicker, animated: true)
    }
    
    func presentPhotoPicker(sourceType: UIImagePickerController.SourceType) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = sourceType
        present(picker, animated: true)
    }
}

extension ImageClassificationViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    // MARK: - Handling Image Picker Selection

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        let info = convertFromUIImagePickerControllerInfoKeyDictionary(info)

        picker.dismiss(animated: true)
        
        let image = info[convertFromUIImagePickerControllerInfoKey(UIImagePickerController.InfoKey.originalImage)] as! UIImage
        
        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        let attentionRequest = VNImageRequestHandler(cgImage: image.cgImage!, orientation: orientation, options: [:])
        
        try? attentionRequest.perform([VNGenerateAttentionBasedSaliencyImageRequest()])
        
        // TODO: - Add logic to choose whether to use detected attention image or the original image to processing.
        imageView.image = image
        updateClassifications(for: image)
    }
}

fileprivate func convertFromUIImagePickerControllerInfoKeyDictionary(_ input: [UIImagePickerController.InfoKey: Any]) -> [String: Any] {
	return Dictionary(uniqueKeysWithValues: input.map {key, value in (key.rawValue, value)})
}

fileprivate func convertFromUIImagePickerControllerInfoKey(_ input: UIImagePickerController.InfoKey) -> String {
	return input.rawValue
}
