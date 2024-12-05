import 'package:flutter/material.dart';

class CustomDropdown extends StatelessWidget {
  final String label;
  final List<String> items;
  final ValueNotifier<String?> selectedValue;
  final String? Function(String?)? validator;
  final void Function(String?)? onChanged;
  final Color? dropdownBackgroundColor; // Add this line

  const CustomDropdown({
    Key? key,
    required this.label,
    required this.items,
    required this.selectedValue,
    this.validator,
    this.onChanged,
    this.dropdownBackgroundColor, // Add this line
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: ValueListenableBuilder<String?>(
        valueListenable: selectedValue,
        builder: (context, value, _) {
          return Container(
            decoration: BoxDecoration(
              color: dropdownBackgroundColor ?? Colors.white, // Set background color
              borderRadius: BorderRadius.circular(8.0),
            ),
            child: DropdownButtonFormField<String>(
              value: value,
              decoration: InputDecoration(
                labelText: label,
                labelStyle: TextStyle(color: Colors.teal),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8.0),
                  borderSide: BorderSide(color: Colors.teal, width: 2.0),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8.0),
                  borderSide: BorderSide(color: Colors.tealAccent, width: 2.0),
                ),
              ),
              icon: Icon(Icons.arrow_drop_down, color: Colors.teal),
              items: items
                  .map((item) => DropdownMenuItem(
                        value: item,
                        child: Text(item),
                      ))
                  .toList(),
              onChanged: (newValue) {
                selectedValue.value = newValue;
                if (onChanged != null) {
                  onChanged!(newValue);
                }
              },
              validator: validator,
            ),
          );
        },
      ),
    );
  }
}
