"""Compute the six selected metrics for saved agent-connectivity catalogs."""
import html
import json
from pathlib import Path
import networkx as nx

HERE = Path(__file__).resolve().parent


def topology_stats(topology, agent_count):
    graph = nx.Graph()
    graph.add_nodes_from(node['id'] for node in topology['nodes'])
    graph.add_edges_from(topology['edges'])
    if not nx.is_connected(graph):
        raise ValueError('Metrics require a connected topology')
    centrality = nx.betweenness_centrality(graph, normalized=True, weight=None, endpoints=False)
    return dict(
        agent_count=agent_count, topology=topology['id'], name=topology['name'],
        total_nodes=graph.number_of_nodes(), edges=graph.number_of_edges(),
        average_shortest_path_length=nx.average_shortest_path_length(graph),
        diameter=nx.diameter(graph), density=nx.density(graph),
        mean_betweenness_centrality=sum(centrality.values()) / len(graph),
        clustering_coefficient=nx.average_clustering(graph, count_zeros=True),
        node_metrics={v: {'betweenness_centrality': b} for v, b in centrality.items()})


def main():
    rows = []
    for path in sorted(HERE.glob('agents_*_topologies.json')):
        catalog = json.loads(path.read_text())
        rows.extend(topology_stats(t, catalog['agent_count']) for t in catalog['topologies'])
    assert len(rows) == 153
    for r in rows:
        n = r['total_nodes']
        assert abs(r['mean_betweenness_centrality'] - (r['average_shortest_path_length']-1)/(n-2)) < 1e-10
        if r['topology'] == 'mesh':
            assert r['diameter'] == 1 and r['density'] == 1
            assert r['mean_betweenness_centrality'] == 0 and r['clustering_coefficient'] == 1
        if r['topology'] == 'star':
            assert abs(r['average_shortest_path_length']-2*(n-1)/n) < 1e-12
            assert r['clustering_coefficient'] == 0
    data = dict(networkx_version=nx.__version__, scope='Connected undirected unweighted graphs; all nodes and edges include routing/supervisory elements. Betweenness is normalized and excludes endpoints. Average betweenness and clustering average over every node, including routing nodes. Clustering includes degree<2 zeros.', rows=rows)
    (HERE/'topology_stats.json').write_text(json.dumps(data, indent=2)+'\n')
    columns = [('agent_count','Agents'),('name','Topology'),('total_nodes','All nodes'),('edges','Edges'),
               ('average_shortest_path_length','Average Shortest Path'),('diameter','Diameter'),
               ('density','Density'),('mean_betweenness_centrality','Average Betweenness Centrality'),
               ('clustering_coefficient','Clustering Coefficient')]
    table = []
    for r in rows:
        cells = ''.join('<td>'+html.escape(f'{r[k]:.5f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k,_ in columns)
        nodes = '<details><summary>Node values</summary>' + '<br>'.join(f'{html.escape(v)}: {values["betweenness_centrality"]:.5f}' for v,values in r['node_metrics'].items()) + '</details>'
        table.append(f'<tr data-agents="{r["agent_count"]}">{cells}<td>{nodes}</td></tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Topology statistics</title>
<style>body{font:15px system-ui;margin:32px;color:#20242a}table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:right;border-bottom:1px solid #ddd;vertical-align:top}th{position:sticky;top:0;background:#f3f4f6}td:nth-child(2),td:last-child{text-align:left}select{font:inherit;padding:6px}summary{cursor:pointer}</style>
<h1>Agent topology statistics</h1><p>153 combinations · 4–20 task agents · nine topologies per size. Source: saved seed-42 topology catalogs.</p>
<p>All routing/supervisory nodes and edges are included. Distances are in hops. Normalized node betweenness excludes endpoints. Both averages include every node; clustering includes zeros for degree below two. Agents, total nodes and edges are descriptive metadata.</p>
<label>Task agents <select id="agents"><option value="all">All</option>''' + ''.join(f'<option>{n}</option>' for n in range(4,21)) + '''</select></label>
<p><a href="topology_stats_formulas.md">Formulas and conventions</a> · <a href="topology_stats.json">Full-precision JSON</a></p><table><thead><tr>''' + ''.join('<th>'+label+'</th>' for _,label in columns) + '<th>Betweenness Centrality</th></tr></thead><tbody>' + ''.join(table) + '''</tbody></table><script>document.getElementById('agents').onchange=function(){for(const row of document.querySelectorAll('tbody tr'))row.hidden=this.value!=='all'&&row.dataset.agents!==this.value;};</script></html>'''
    (HERE/'topology_stats.html').write_text(page)
    print('Updated 153 combinations with only the six selected metrics; validation passed.')


if __name__ == '__main__':
    main()
