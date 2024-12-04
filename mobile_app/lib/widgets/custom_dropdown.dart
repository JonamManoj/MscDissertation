import 'package:flutter/material.dart';

class CustomDropdown extends StatelessWidget {
  final String label;
  final List<String> items;
  final void Function(String?)? onSaved;

  const CustomDropdown({
    Key? key,
    required this.label,
    required this.items,
    this.onSaved,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: DropdownButtonFormField<String>(
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
        items: items
            .map((item) => DropdownMenuItem(
                  value: item,
                  child: Text(item),
                ))
            .toList(),
        onChanged: (value) {},
        validator: (value) {
          if (value == null) {
            return 'Please select $label';
          }
          return null;
        },
        onSaved: onSaved,
      ),
    );
  }
}
