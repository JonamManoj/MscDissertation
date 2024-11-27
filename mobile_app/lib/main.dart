import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(CKDApp());
}

class CKDApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CKD Predictor',
      home: CKDForm(),
    );
  }
}

class CKDForm extends StatefulWidget {
  @override
  _CKDFormState createState() => _CKDFormState();
}

class _CKDFormState extends State<CKDForm> {
  final _formKey = GlobalKey<FormState>();
  final Map<String, dynamic> inputData = {};

  String predictionResult = '';
  String selectedModel = 'best_ckd_model';

  Future<void> makePrediction() async {
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();
      
      // Prepare input in the format expected by the API
      Map<String, dynamic> input = {
        "input": [
          inputData['age'],
          inputData['bp'],
          inputData['sg'],
          inputData['al'],
          inputData['su'],
          inputData['rbc'],
          inputData['pc'],
          inputData['pcc'],
          inputData['ba'],
          inputData['bgr'],
          inputData['bu'],
          inputData['sc'],
          inputData['sod'],
          inputData['pot'],
          inputData['hemo'],
          inputData['pcv'],
          inputData['wbcc'],
          inputData['rbcc'],
          inputData['htn'],
          inputData['dm'],
          inputData['cad'],
          inputData['appet'],
          inputData['pe'],
          inputData['ane'],
        ],
        "model": selectedModel,
      };

      try {
        final response = await http.post(
          Uri.parse('http://172.16.2.55:9999/predict'),
          headers: <String, String>{
            'Content-Type': 'application/json',
          },
          body: jsonEncode(input),
        );

        if (response.statusCode == 200) {
          var data = jsonDecode(response.body);
          setState(() {
            predictionResult = data['prediction'];
          });
        } else {
          setState(() {
            predictionResult = 'Failed to get prediction. Try again!';
          });
        }
      } catch (e) {
        setState(() {
          predictionResult = 'Error: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('CKD Predictor'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              buildTextField('Age', 'age'),
              buildTextField('Blood Pressure (bp)', 'bp'),
              buildTextField('Specific Gravity (sg)', 'sg'),
              buildTextField('Albumin (al)', 'al'),
              buildTextField('Sugar (su)', 'su'),
              buildDropdown('Red Blood Cells (rbc)', 'rbc', ['normal', 'abnormal']),
              buildDropdown('Pus Cell (pc)', 'pc', ['normal', 'abnormal']),
              buildDropdown('Pus Cell Clumps (pcc)', 'pcc', ['present', 'notpresent']),
              buildDropdown('Bacteria (ba)', 'ba', ['present', 'notpresent']),
              buildTextField('Blood Glucose Random (bgr)', 'bgr'),
              buildTextField('Blood Urea (bu)', 'bu'),
              buildTextField('Serum Creatinine (sc)', 'sc'),
              buildTextField('Sodium (sod)', 'sod'),
              buildTextField('Potassium (pot)', 'pot'),
              buildTextField('Hemoglobin (hemo)', 'hemo'),
              buildTextField('Packed Cell Volume (pcv)', 'pcv'),
              buildTextField('White Blood Cell Count (wbcc)', 'wbcc'),
              buildTextField('Red Blood Cell Count (rbcc)', 'rbcc'),
              buildDropdown('Hypertension (htn)', 'htn', ['yes', 'no']),
              buildDropdown('Diabetes Mellitus (dm)', 'dm', ['yes', 'no']),
              buildDropdown('Coronary Artery Disease (cad)', 'cad', ['yes', 'no']),
              buildDropdown('Appetite (appet)', 'appet', ['good', 'poor']),
              buildDropdown('Pedal Edema (pe)', 'pe', ['yes', 'no']),
              buildDropdown('Anemia (ane)', 'ane', ['yes', 'no']),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: makePrediction,
                child: Text('Predict CKD'),
              ),
              SizedBox(height: 20),
              Text(
                'Prediction: $predictionResult',
                style: TextStyle(fontSize: 20),
              ),
            ],
          ),
        ),
      ),
    );
  }

  TextFormField buildTextField(String label, String key) {
    return TextFormField(
      decoration: InputDecoration(labelText: label),
      keyboardType: TextInputType.number,
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter $label';
        }
        return null;
      },
      onSaved: (value) {
        inputData[key] = double.parse(value!);
      },
    );
  }

  DropdownButtonFormField<String> buildDropdown(String label, String key, List<String> items) {
    return DropdownButtonFormField<String>(
      decoration: InputDecoration(labelText: label),
      items: items.map((item) {
        return DropdownMenuItem(
          value: item,
          child: Text(item),
        );
      }).toList(),
      onChanged: (value) {
        setState(() {
          inputData[key] = value;
        });
      },
      validator: (value) {
        if (value == null) {
          return 'Please select $label';
        }
        return null;
      },
      onSaved: (value) {
        inputData[key] = value;
      },
    );
  }
}
