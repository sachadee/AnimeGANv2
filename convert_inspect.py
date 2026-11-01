import tensorflow as tf
import subprocess

checkpoint_meta = "/content/AnimeGANv2/checkpoint/generator_indyStyle_weight/indyStyle-50.ckpt.meta"

sess = tf.Session()
saver = tf.train.import_meta_graph(checkpoint_meta, clear_devices=True)
graph = tf.get_default_graph()

nodes = [n.name for n in graph.as_graph_def().node]
print(f"Total de nós no grafo: {len(nodes)}")

# Filtrar potenciais candidatos de saída pela operação Tanh ou escopo G_net
tanh_nodes = [name for name in nodes if 'tanh' in name.lower() or 'fake' in name.lower() or 'g_net' in name.lower()]
print("Candidatos de saída encontrados:", tanh_nodes[-10:])

# Vamos buscar o último nó operacional do gerador
# Geralmente no AnimeGANv2, a saída após get_generator_ckpt é "generator/G_net/fake" ou apenas "G_net/fake" 
# Se as estruturas acima não forem encontradas, pegamos o último nó que tenha relação com o gerador ou Tanh
output_node = ""
for candidate in ["generator/G_net/fake", "G_net/fake", "generator/G_net/out", "G_net/out"]:
    if candidate in nodes:
        output_node = candidate
        break

if not output_node:
    # Se não achar por nome conhecido, tenta pegar o último Tanh correspondente à saída
    tanhas = [name for name in nodes if 'tanh' in name.lower()]
    if tanhas:
        output_node = tanhas[-1]
    else:
        # Fallback
        output_node = nodes[-1]

input_node = "generator_input"
print(f"\nNó de Entrada Selecionado: {input_node}:0")
print(f"Nó de Saída Selecionado: {output_node}:0")

cmd = [
    "python", "-m", "tf2onnx.convert",
    "--checkpoint", checkpoint_meta,
    "--output", "/content/AnimeGANv2/AnimeGANv2_indyStyle_50.onnx",
    "--inputs", f"{input_node}:0",
    "--outputs", f"{output_node}:0",
    "--opset", "11"
]

subprocess.run(cmd)

