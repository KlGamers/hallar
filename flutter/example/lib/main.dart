import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hallar/hallar.dart';
import 'package:path_provider/path_provider.dart';

const docs = {
  '0': 'el perro y el gato',
  '1': 'cafe con leche caliente',
  '2': 'receta de croquetas de jamon',
  '3': 'boleto de tren para el viaje',
};

Future<HallarIndex> load() async {
  final raw = await rootBundle.load('assets/tiny.hallar');
  final dir = await getApplicationSupportDirectory();
  final file = File('${dir.path}/tiny.hallar')..writeAsBytesSync(raw.buffer.asUint8List());
  return HallarIndex(HallarModel.open(file.path))..addAll(docs);
}

void main() => runApp(const App());

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) => const MaterialApp(home: SearchPage());
}

class SearchPage extends StatefulWidget {
  const SearchPage({super.key});

  @override
  State<SearchPage> createState() => _SearchPageState();
}

class _SearchPageState extends State<SearchPage> {
  HallarIndex? idx;
  List<SearchResult> hits = const [];

  @override
  void initState() {
    super.initState();
    load().then((v) => setState(() => idx = v));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('hallar')),
      body: idx == null
          ? const Center(child: CircularProgressIndicator())
          : Column(children: [
              Padding(
                padding: const EdgeInsets.all(16),
                child: TextField(
                  decoration: const InputDecoration(hintText: 'Buscar sin internet'),
                  onChanged: (q) => setState(() => hits = q.isEmpty ? const [] : idx!.search(q, limit: 5)),
                ),
              ),
              Expanded(
                child: ListView(children: [for (final h in hits) ListTile(title: Text(h.text))]),
              ),
            ]),
    );
  }
}
