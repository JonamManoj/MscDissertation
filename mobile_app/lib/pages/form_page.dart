
// import 'package:flutter/material.dart';
// import '../widgets/custom_text_field.dart';
// import '../widgets/custom_dropdown.dart';
// import 'result_page.dart';
// import 'package:http/http.dart' as http;
// import 'dart:convert';

// class FormPage extends StatefulWidget {
//   const FormPage({Key? key}) : super(key: key);

//   @override
//   State<FormPage> createState() => _FormPageState();
// }

// class _FormPageState extends State<FormPage> {
//   final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

//   // Controllers for text fields
//   final TextEditingController ageController = TextEditingController();
//   final TextEditingController bpController = TextEditingController();
//   final TextEditingController sgController = TextEditingController();
//   final TextEditingController alController = TextEditingController();
//   final TextEditingController suController = TextEditingController();
//   final TextEditingController bgrController = TextEditingController();
//   final TextEditingController buController = TextEditingController();
//   final TextEditingController scController = TextEditingController();
//   final TextEditingController sodController = TextEditingController();
//   final TextEditingController potController = TextEditingController();
//   final TextEditingController hemoController = TextEditingController();
//   final TextEditingController pcvController = TextEditingController();
//   final TextEditingController wbccController = TextEditingController();
//   final TextEditingController rbccController = TextEditingController();

//   // ValueNotifiers for dropdowns
//   final ValueNotifier<String?> rbcNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> pcNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> pccNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> baNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> htnNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> dmNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> cadNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> appetNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> peNotifier = ValueNotifier(null);
//   final ValueNotifier<String?> aneNotifier = ValueNotifier(null);

//   void _submitForm() async {
//     if (!_formKey.currentState!.validate()) {
//       print('Form validation failed');
//       _showErrorDialog(context, 'Please fill in all required fields correctly.');
//       return;
//     }

//     print('Form validation passed');

//     // Prepare input data
//     final inputData = {
//       "age": int.tryParse(ageController.text),
//       "bp": int.tryParse(bpController.text),
//       "sg": double.tryParse(sgController.text),
//       "al": int.tryParse(alController.text),
//       "su": int.tryParse(suController.text),
//       "rbc": rbcNotifier.value,
//       "pc": pcNotifier.value,
//       "pcc": pccNotifier.value,
//       "ba": baNotifier.value,
//       "bgr": int.tryParse(bgrController.text),
//       "bu": int.tryParse(buController.text),
//       "sc": double.tryParse(scController.text),
//       "sod": double.tryParse(sodController.text),
//       "pot": double.tryParse(potController.text),
//       "hemo": double.tryParse(hemoController.text),
//       "pcv": int.tryParse(pcvController.text),
//       "wbcc": int.tryParse(wbccController.text),
//       "rbcc": double.tryParse(rbccController.text),
//       "htn": htnNotifier.value,
//       "dm": dmNotifier.value,
//       "cad": cadNotifier.value,
//       "appet": appetNotifier.value,
//       "pe": peNotifier.value,
//       "ane": aneNotifier.value,
//     };

//     print('Captured Input Data: $inputData');

//     try {
//       // Show loading indicator while making request
//       showDialog(
//         context: context,
//         barrierDismissible: false,
//         builder: (BuildContext context) {
//           return const Center(
//             child: CircularProgressIndicator(),
//           );
//         },
//       );

//       final response = await http.post(
//         Uri.parse('http://172.16.0.82:8080/diagnosis'),
//         headers: {'Content-Type': 'application/json'},
//         body: jsonEncode({
//           "patient_data": inputData,
//           "model_name": "Random Forest" // Default or modify according to your requirement
//         }),
//       );

//       Navigator.pop(context); // Remove loading indicator

//       print('Response Status Code: ${response.statusCode}');
//       print('Response Body: ${response.body}');

//       if (response.statusCode == 200) {
//         final data = jsonDecode(response.body);

//         // Extract confidence probabilities
//         List<dynamic> confidenceList = data['confidence'] ?? [];
//         double confidenceCKD = 0.0;
//         double confidenceNoCKD = 0.0;

//         if (confidenceList.isNotEmpty && confidenceList[0] is double) {
//           confidenceCKD = (confidenceList[1] as num).toDouble();
//           confidenceNoCKD = (confidenceList[0] as num).toDouble();
//         }

//         print('Confidence CKD: $confidenceCKD');
//         print('Confidence No CKD: $confidenceNoCKD');

//         Navigator.push(
//           context,
//           MaterialPageRoute(
//             builder: (context) => ResultPage(
//               diagnosis: data['diagnosis'] ?? "No diagnosis available",
//               confidenceCKD: confidenceCKD, 
//               confidenceNoCKD: confidenceNoCKD, 
//             ),
//           ),
//         );
//       } else {
//         _showErrorDialog(context, 'Failed to get prediction. Try again!');
//       }
//     } catch (e) {
//       Navigator.pop(context); // Remove loading indicator if an error occurs
//       _showErrorDialog(context, 'Error: $e');
//     }
//   }


//   void _showErrorDialog(BuildContext context, String message) {
//     showDialog(
//       context: context,
//       builder: (BuildContext context) {
//         return AlertDialog(
//           title: const Text("Error"),
//           content: Text(message),
//           actions: [
//             TextButton(
//               onPressed: () => Navigator.of(context).pop(),
//               child: const Text("OK"),
//             ),
//           ],
//         );
//       },
//     );
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: const Text('CKD Analyser Form'),
//       ),
//       body: Padding(
//         padding: const EdgeInsets.all(16.0),
//         child: Form(
//           key: _formKey,
//           child: ListView(
//             children: [
//               CustomTextField(
//                 label: 'Age',
//                 controller: ageController,
//                 validator: (value) => value == null || value.isEmpty ? 'Age cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Blood Pressure (BP)',
//                 controller: bpController,
//                 validator: (value) => value == null || value.isEmpty ? 'BP cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Specific Gravity (SG)',
//                 controller: sgController,
//                 validator: (value) => value == null || value.isEmpty ? 'SG cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Albumin (AL)',
//                 controller: alController,
//                 validator: (value) => value == null || value.isEmpty ? 'AL cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Sugar (SU)',
//                 controller: suController,
//                 validator: (value) => value == null || value.isEmpty ? 'SU cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Red Blood Cells (RBC)',
//                 items: ['normal', 'abnormal'],
//                 selectedValue: rbcNotifier,
//                 validator: (value) => value == null ? 'RBC cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Pus Cell (PC)',
//                 items: ['normal', 'abnormal'],
//                 selectedValue: pcNotifier,
//                 validator: (value) => value == null ? 'PC cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Pus Cell Clumps (PCC)',
//                 items: ['present', 'notpresent'],
//                 selectedValue: pccNotifier,
//                 validator: (value) => value == null ? 'PCC cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Bacteria (BA)',
//                 items: ['present', 'notpresent'],
//                 selectedValue: baNotifier,
//                 validator: (value) => value == null ? 'BA cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Blood Glucose Random (BGR)',
//                 controller: bgrController,
//                 validator: (value) => value == null || value.isEmpty ? 'BGR cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Blood Urea (BU)',
//                 controller: buController,
//                 validator: (value) => value == null || value.isEmpty ? 'BU cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Serum Creatinine (SC)',
//                 controller: scController,
//                 validator: (value) => value == null || value.isEmpty ? 'SC cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Sodium (SOD)',
//                 controller: sodController,
//                 validator: (value) => value == null || value.isEmpty ? 'SOD cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Potassium (POT)',
//                 controller: potController,
//                 validator: (value) => value == null || value.isEmpty ? 'POT cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Hemoglobin (HEMO)',
//                 controller: hemoController,
//                 validator: (value) => value == null || value.isEmpty ? 'HEMO cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Packed Cell Volume (PCV)',
//                 controller: pcvController,
//                 validator: (value) => value == null || value.isEmpty ? 'PCV cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'White Blood Cell Count (WBCC)',
//                 controller: wbccController,
//                 validator: (value) => value == null || value.isEmpty ? 'WBCC cannot be empty' : null,
//               ),
//               CustomTextField(
//                 label: 'Red Blood Cell Count (RBCC)',
//                 controller: rbccController,
//                 validator: (value) => value == null || value.isEmpty ? 'RBCC cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Hypertension (HTN)',
//                 items: ['yes', 'no'],
//                 selectedValue: htnNotifier,
//                 validator: (value) => value == null ? 'HTN cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Diabetes Mellitus (DM)',
//                 items: ['yes', 'no'],
//                 selectedValue: dmNotifier,
//                 validator: (value) => value == null ? 'DM cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Coronary Artery Disease (CAD)',
//                 items: ['yes', 'no'],
//                 selectedValue: cadNotifier,
//                 validator: (value) => value == null ? 'CAD cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Appetite (APPET)',
//                 items: ['good', 'poor'],
//                 selectedValue: appetNotifier,
//                 validator: (value) => value == null ? 'Appetite cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Pedal Edema (PE)',
//                 items: ['yes', 'no'],
//                 selectedValue: peNotifier,
//                 validator: (value) => value == null ? 'PE cannot be empty' : null,
//               ),
//               CustomDropdown(
//                 label: 'Anemia (ANE)',
//                 items: ['yes', 'no'],
//                 selectedValue: aneNotifier,
//                 validator: (value) => value == null ? 'ANE cannot be empty' : null,
//               ),
//               const SizedBox(height: 20),
//               ElevatedButton(
//                 onPressed: _submitForm,
//                 child: const Text('Submit'),
//               ),
//             ],
//           ),
//         ),
//       ),
//     );
//   }
// }

import 'package:flutter/material.dart';
import '../widgets/custom_text_field.dart';
import '../widgets/custom_dropdown.dart';
import 'result_page.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class FormPage extends StatefulWidget {
  const FormPage({Key? key}) : super(key: key);

  @override
  State<FormPage> createState() => _FormPageState();
}

class _FormPageState extends State<FormPage> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  // Controllers for text fields
  final TextEditingController ageController = TextEditingController();
  final TextEditingController bpController = TextEditingController();
  final TextEditingController sgController = TextEditingController();
  final TextEditingController alController = TextEditingController();
  final TextEditingController suController = TextEditingController();
  final TextEditingController bgrController = TextEditingController();
  final TextEditingController buController = TextEditingController();
  final TextEditingController scController = TextEditingController();
  final TextEditingController sodController = TextEditingController();
  final TextEditingController potController = TextEditingController();
  final TextEditingController hemoController = TextEditingController();
  final TextEditingController pcvController = TextEditingController();
  final TextEditingController wbccController = TextEditingController();
  final TextEditingController rbccController = TextEditingController();

  // ValueNotifiers for dropdowns
  final ValueNotifier<String?> rbcNotifier = ValueNotifier(null);
  final ValueNotifier<String?> pcNotifier = ValueNotifier(null);
  final ValueNotifier<String?> pccNotifier = ValueNotifier(null);
  final ValueNotifier<String?> baNotifier = ValueNotifier(null);
  final ValueNotifier<String?> htnNotifier = ValueNotifier(null);
  final ValueNotifier<String?> dmNotifier = ValueNotifier(null);
  final ValueNotifier<String?> cadNotifier = ValueNotifier(null);
  final ValueNotifier<String?> appetNotifier = ValueNotifier(null);
  final ValueNotifier<String?> peNotifier = ValueNotifier(null);
  final ValueNotifier<String?> aneNotifier = ValueNotifier(null);

  // Model Selection
  String? selectedModel;

  final List<String> models = [
    'Logistic Regression', 'SVM', 'Random Forest', 'XGBoost', 'ANN', 'CNN', 'RNN', 'LSTM'
  ];

  void _submitForm() async {
    if (!_formKey.currentState!.validate()) {
      print('Form validation failed');
      _showErrorDialog(context, 'Please fill in all required fields correctly.');
      return;
    }

    if (selectedModel == null) {
      _showErrorDialog(context, 'Please select a model for the prediction.');
      return;
    }

    print('Form validation passed');

    // Prepare input data
    final inputData = {
      "age": int.tryParse(ageController.text),
      "bp": int.tryParse(bpController.text),
      "sg": double.tryParse(sgController.text),
      "al": int.tryParse(alController.text),
      "su": int.tryParse(suController.text),
      "rbc": rbcNotifier.value,
      "pc": pcNotifier.value,
      "pcc": pccNotifier.value,
      "ba": baNotifier.value,
      "bgr": int.tryParse(bgrController.text),
      "bu": int.tryParse(buController.text),
      "sc": double.tryParse(scController.text),
      "sod": double.tryParse(sodController.text),
      "pot": double.tryParse(potController.text),
      "hemo": double.tryParse(hemoController.text),
      "pcv": int.tryParse(pcvController.text),
      "wbcc": int.tryParse(wbccController.text),
      "rbcc": double.tryParse(rbccController.text),
      "htn": htnNotifier.value,
      "dm": dmNotifier.value,
      "cad": cadNotifier.value,
      "appet": appetNotifier.value,
      "pe": peNotifier.value,
      "ane": aneNotifier.value,
    };

    print('Captured Input Data: $inputData');

    try {
      // Show loading indicator while making request
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext context) {
          return const Center(
            child: CircularProgressIndicator(),
          );
        },
      );

      final response = await http.post(
        Uri.parse('http://172.16.0.82:8080/diagnosis'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "patient_data": inputData,
          "model_name": selectedModel, // Use selected model
        }),
      );

      Navigator.pop(context); // Remove loading indicator

      print('Response Status Code: ${response.statusCode}');
      print('Response Body: ${response.body}');

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        // Extract confidence probabilities
        List<dynamic> confidenceList = data['confidence'] ?? [];
        double confidenceCKD = 0.0;
        double confidenceNoCKD = 0.0;

        if (confidenceList.isNotEmpty && confidenceList[0] is double) {
          confidenceCKD = (confidenceList[1] as num).toDouble();
          confidenceNoCKD = (confidenceList[0] as num).toDouble();
        }

        print('Confidence CKD: $confidenceCKD');
        print('Confidence No CKD: $confidenceNoCKD');

        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultPage(
              diagnosis: data['diagnosis'] ?? "No diagnosis available",
              confidenceCKD: confidenceCKD,
              confidenceNoCKD: confidenceNoCKD,
              selectedModel: selectedModel!,
            ),
          ),
        );
      } else {
        _showErrorDialog(context, 'Failed to get prediction. Try again!');
      }
    } catch (e) {
      Navigator.pop(context); // Remove loading indicator if an error occurs
      _showErrorDialog(context, 'Error: $e');
    }
  }

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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('CKD Analyser Form'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              CustomTextField(
                label: 'Age',
                controller: ageController,
                validator: (value) => value == null || value.isEmpty ? 'Age cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Blood Pressure (BP)',
                controller: bpController,
                validator: (value) => value == null || value.isEmpty ? 'BP cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Specific Gravity (SG)',
                controller: sgController,
                validator: (value) => value == null || value.isEmpty ? 'SG cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Albumin (AL)',
                controller: alController,
                validator: (value) => value == null || value.isEmpty ? 'AL cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Sugar (SU)',
                controller: suController,
                validator: (value) => value == null || value.isEmpty ? 'SU cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Red Blood Cells (RBC)',
                items: ['normal', 'abnormal'],
                selectedValue: rbcNotifier,
                validator: (value) => value == null ? 'RBC cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Pus Cell (PC)',
                items: ['normal', 'abnormal'],
                selectedValue: pcNotifier,
                validator: (value) => value == null ? 'PC cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Pus Cell Clumps (PCC)',
                items: ['present', 'notpresent'],
                selectedValue: pccNotifier,
                validator: (value) => value == null ? 'PCC cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Bacteria (BA)',
                items: ['present', 'notpresent'],
                selectedValue: baNotifier,
                validator: (value) => value == null ? 'BA cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Blood Glucose Random (BGR)',
                controller: bgrController,
                validator: (value) => value == null || value.isEmpty ? 'BGR cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Blood Urea (BU)',
                controller: buController,
                validator: (value) => value == null || value.isEmpty ? 'BU cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Serum Creatinine (SC)',
                controller: scController,
                validator: (value) => value == null || value.isEmpty ? 'SC cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Sodium (SOD)',
                controller: sodController,
                validator: (value) => value == null || value.isEmpty ? 'SOD cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Potassium (POT)',
                controller: potController,
                validator: (value) => value == null || value.isEmpty ? 'POT cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Hemoglobin (HEMO)',
                controller: hemoController,
                validator: (value) => value == null || value.isEmpty ? 'HEMO cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Packed Cell Volume (PCV)',
                controller: pcvController,
                validator: (value) => value == null || value.isEmpty ? 'PCV cannot be empty' : null,
              ),
              CustomTextField(
                label: 'White Blood Cell Count (WBCC)',
                controller: wbccController,
                validator: (value) => value == null || value.isEmpty ? 'WBCC cannot be empty' : null,
              ),
              CustomTextField(
                label: 'Red Blood Cell Count (RBCC)',
                controller: rbccController,
                validator: (value) => value == null || value.isEmpty ? 'RBCC cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Hypertension (HTN)',
                items: ['yes', 'no'],
                selectedValue: htnNotifier,
                validator: (value) => value == null ? 'HTN cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Diabetes Mellitus (DM)',
                items: ['yes', 'no'],
                selectedValue: dmNotifier,
                validator: (value) => value == null ? 'DM cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Coronary Artery Disease (CAD)',
                items: ['yes', 'no'],
                selectedValue: cadNotifier,
                validator: (value) => value == null ? 'CAD cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Appetite (APPET)',
                items: ['good', 'poor'],
                selectedValue: appetNotifier,
                validator: (value) => value == null ? 'Appetite cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Pedal Edema (PE)',
                items: ['yes', 'no'],
                selectedValue: peNotifier,
                validator: (value) => value == null ? 'PE cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Anemia (ANE)',
                items: ['yes', 'no'],
                selectedValue: aneNotifier,
                validator: (value) => value == null ? 'ANE cannot be empty' : null,
              ),
              CustomDropdown(
                label: 'Model Selection',
                items: models,
                selectedValue: ValueNotifier(selectedModel),
                validator: (value) => value == null ? 'Please select a model' : null,
                onChanged: (newValue) {
                  setState(() {
                    selectedModel = newValue;
                  });
                },
                dropdownBackgroundColor: Colors.deepPurple[100], // Set background color to violet
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
