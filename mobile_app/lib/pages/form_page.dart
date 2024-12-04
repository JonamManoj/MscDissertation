import 'package:flutter/material.dart';
import '../widgets/custom_text_field.dart';
import '../widgets/custom_dropdown.dart';
import 'result_page.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class FormPage extends StatelessWidget {
  const FormPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final GlobalKey<FormState> _formKey = GlobalKey<FormState>();
    final Map<String, dynamic> inputData = {};

    void _showErrorDialog(BuildContext context, String message) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text("Error"),
            content: Text(message),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text("OK"),
              ),
            ],
          );
        },
      );
    }

    void _submitForm() async {
      if (!_formKey.currentState!.validate()) {
        return; // Stop if the form is invalid
      }

      _formKey.currentState!.save();

      print('Captured Input Data: $inputData');

      try {
        // Prepare input for API
        Map<String, dynamic> input = {
          "patient_data": {
            "age": inputData['age'] != null ? double.tryParse(inputData['age']!) : null,
            "bp": inputData['bp'] != null ? double.tryParse(inputData['bp']!) : null,
            "sg": inputData['sg'] != null ? double.tryParse(inputData['sg']!) : null,
            "al": inputData['al'] != null ? int.tryParse(inputData['al']!) : null,
            "su": inputData['su'] != null ? int.tryParse(inputData['su']!) : null,
            "rbc": inputData['rbc'],
            "pc": inputData['pc'],
            "pcc": inputData['pcc'],
            "ba": inputData['ba'],
            "bgr": inputData['bgr'] != null ? double.tryParse(inputData['bgr']!) : null,
            "bu": inputData['bu'] != null ? double.tryParse(inputData['bu']!) : null,
            "sc": inputData['sc'] != null ? double.tryParse(inputData['sc']!) : null,
            "sod": inputData['sod'] != null ? double.tryParse(inputData['sod']!) : null,
            "pot": inputData['pot'] != null ? double.tryParse(inputData['pot']!) : null,
            "hemo": inputData['hemo'] != null ? double.tryParse(inputData['hemo']!) : null,
            "pcv": inputData['pcv'] != null ? int.tryParse(inputData['pcv']!) : null,
            "wbcc": inputData['wbcc'] != null ? int.tryParse(inputData['wbcc']!) : null,
            "rbcc": inputData['rbcc'] != null ? double.tryParse(inputData['rbcc']!) : null,
            "htn": inputData['htn'],
            "dm": inputData['dm'],
            "cad": inputData['cad'],
            "appet": inputData['appet'],
            "pe": inputData['pe'],
            "ane": inputData['ane'],
          },
          "model": "best_ckd_model",
        };

        print('API Request Body: $input');

        // Make the API request
        final response = await http.post(
          Uri.parse('http://172.16.2.86:8080/diagnosis'),
          headers: <String, String>{'Content-Type': 'application/json'},
          body: jsonEncode(input),
        );

        print('Response Status Code: ${response.statusCode}');
        print('Response Body: ${response.body}');

        if (response.statusCode == 200) {
          var data = jsonDecode(response.body);

          // Extract confidence
          List<dynamic> confidenceList = data['confidence'] ?? [];
          List<double> confidence = confidenceList.isNotEmpty ? confidenceList[0].cast<double>() : [0.0, 0.0];

          double confidenceCKD = confidence.length > 1 ? confidence[1] : 0.0;
          double confidenceNoCKD = confidence.length > 0 ? confidence[0] : 0.0;

          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultPage(
                diagnosis: data['diagnosis'] ?? "No diagnosis available",
                confidenceCKD: confidenceCKD,
                confidenceNoCKD: confidenceNoCKD,
              ),
            ),
          );
        } else {
          _showErrorDialog(context, 'Failed to get prediction. Try again!');
        }
      } catch (e) {
        _showErrorDialog(context, 'Error: $e');
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('CKD Predictor Form'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              CustomTextField(
                  label: 'Age', onSaved: (value) => inputData['age'] = value),
              CustomTextField(
                  label: 'Blood Pressure (BP)',
                  onSaved: (value) => inputData['bp'] = value),
              CustomTextField(
                  label: 'Specific Gravity (SG)',
                  onSaved: (value) => inputData['sg'] = value),
              CustomTextField(
                  label: 'Albumin (AL)', onSaved: (value) => inputData['al'] = value),
              CustomTextField(
                  label: 'Sugar (SU)', onSaved: (value) => inputData['su'] = value),
              CustomDropdown(
                label: 'Red Blood Cells (RBC)',
                items: ['normal', 'abnormal'],
                onSaved: (value) => inputData['rbc'] = value,
              ),
              CustomDropdown(
                label: 'Pus Cell (PC)',
                items: ['normal', 'abnormal'],
                onSaved: (value) => inputData['pc'] = value,
              ),
              CustomDropdown(
                label: 'Pus Cell Clumps (PCC)',
                items: ['present', 'notpresent'],
                onSaved: (value) => inputData['pcc'] = value,
              ),
              CustomDropdown(
                label: 'Bacteria (BA)',
                items: ['present', 'notpresent'],
                onSaved: (value) => inputData['ba'] = value,
              ),
              CustomTextField(
                  label: 'Blood Glucose Random (BGR)',
                  onSaved: (value) => inputData['bgr'] = value),
              CustomTextField(
                  label: 'Blood Urea (BU)', onSaved: (value) => inputData['bu'] = value),
              CustomTextField(
                  label: 'Serum Creatinine (SC)',
                  onSaved: (value) => inputData['sc'] = value),
              CustomTextField(
                  label: 'Sodium (SOD)', onSaved: (value) => inputData['sod'] = value),
              CustomTextField(
                  label: 'Potassium (POT)',
                  onSaved: (value) => inputData['pot'] = value),
              CustomTextField(
                  label: 'Hemoglobin (HEMO)',
                  onSaved: (value) => inputData['hemo'] = value),
              CustomTextField(
                  label: 'Packed Cell Volume (PCV)',
                  onSaved: (value) => inputData['pcv'] = value),
              CustomTextField(
                  label: 'White Blood Cell Count (WBCC)',
                  onSaved: (value) => inputData['wbcc'] = value),
              CustomTextField(
                  label: 'Red Blood Cell Count (RBCC)',
                  onSaved: (value) => inputData['rbcc'] = value),
              CustomDropdown(
                label: 'Hypertension (HTN)',
                items: ['yes', 'no'],
                onSaved: (value) => inputData['htn'] = value,
              ),
              CustomDropdown(
                label: 'Diabetes Mellitus (DM)',
                items: ['yes', 'no'],
                onSaved: (value) => inputData['dm'] = value,
              ),
              CustomDropdown(
                label: 'Coronary Artery Disease (CAD)',
                items: ['yes', 'no'],
                onSaved: (value) => inputData['cad'] = value,
              ),
              CustomDropdown(
                label: 'Appetite (APPET)',
                items: ['good', 'poor'],
                onSaved: (value) => inputData['appet'] = value,
              ),
              CustomDropdown(
                label: 'Pedal Edema (PE)',
                items: ['yes', 'no'],
                onSaved: (value) => inputData['pe'] = value,
              ),
              CustomDropdown(
                label: 'Anemia (ANE)',
                items: ['yes', 'no'],
                onSaved: (value) => inputData['ane'] = value,
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _submitForm,
                child: const Text('Submit'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
