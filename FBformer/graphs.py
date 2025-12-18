"""Module to generate networkx graphs."""
"""Implementation based on the template of ALIGNN."""

from re import X
import numpy as np
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes

from jarvis.core.atoms import Atoms
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
import itertools

try:
    import torch
    from tqdm import tqdm
except Exception as exp:
    print("torch/tqdm is not installed.", exp)
    pass

class PygStructureDataset(torch.utils.data.Dataset):


    def __init__(
        self, 
        df: pd.DataFrame, 
        graphs: Sequence[Data], 
        target: str,  
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        classification=False, 
        id_tag="jid",
        neighbor_strategy="", 
        lineControl=True,
        mean_train=None,
        std_train=None,
    ):

        self.df = df
        self.graphs = graphs
        self.target = target
        self.line_graph = line_graph

        self.ids = self.df[id_tag]
        self.atoms = self.df['atoms']
        self.labels = torch.tensor(self.df[target]).type(
            torch.get_default_dtype()
        )
        print("mean %f std %f"%(self.labels.mean(), self.labels.std()))

        if mean_train == None: 
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
            print("normalize using training mean but shall not be used here %f and std %f" % (mean, std))
        else:
            self.labels = (self.labels - mean_train) / std_train 
            print("normalize using training mean %f and std %f" % (mean_train, std_train))

        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        for g in graphs:
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f

        self.prepare_batch = prepare_pyg_batch

        if line_graph:
            self.prepare_batch = prepare_pyg_line_graph_batch
            print("building line graphs")
            if lineControl == False:
                self.line_graphs = []
                self.graphs = []
                for g in tqdm(graphs):
                    linegraph_trans = LineGraph(force_directed=True)
                    g_new = Data()
                    g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
                    try:
                        lg = linegraph_trans(g)
                    except Exception as exp:
                        print(g.x, g.edge_attr, exp)
                        pass
                    lg.edge_attr = pyg_compute_bond_cosines(lg) 
                    # lg.edge_attr = pyg_compute_bond_angle(lg)
                    self.graphs.append(g_new)
                    self.line_graphs.append(lg)
            else:
                if neighbor_strategy == "pairwise-k-nearest":
                    self.graphs = []
                    labels = []
                    idx_t = 0
                    filter_out = 0
                    max_size = 0
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        if g.x.size(0) > max_size:
                            max_size = g.x.size(0)
                        if g.x.size(0) < 200:
                            self.graphs.append(g)
                            labels.append(self.labels[idx_t])
                        else:
                            filter_out += 1
                        idx_t += 1
                    print("filter out %d samples because of exceeding threshold of 200 for nn based method" % filter_out)
                    print("dataset max atom number %d" % max_size)
                    self.line_graphs = self.graphs
                    self.labels = labels
                    self.labels = torch.tensor(self.labels).type(
                                    torch.get_default_dtype()
                                )
                else:
                    self.graphs = []
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        self.graphs.append(g)
                    self.line_graphs = self.graphs

 
        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"): 

        max_z = max(v["Z"] for v in chem_data.values()) 

        template = get_node_attributes("C", atom_features) 

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items(): 
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x   

        return features 

    def __len__(self):

        return self.labels.shape[0]


    def __getitem__(self, idx):

        g = self.graphs[idx]
        label = self.labels[idx] 

        if self.transform: 
            g = self.transform(g)

        if self.line_graph:
            lattice = torch.as_tensor(Atoms.from_dict(self.atoms[idx]).lattice_mat).float()
            return g, self.line_graphs[idx], lattice, label

        return g, label 

    def setup_standardizer(self, ids): 

        x = torch.cat(
            [
                g.x
                for idx, g in enumerate(self.graphs) 
                if idx in ids  
            ]
        )

        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = PygStandardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        graphs, labels = map(list, zip(*samples)) 
        batched_graph = Batch.from_data_list(graphs) 
        return batched_graph, torch.tensor(labels) 

    @staticmethod
    def collate_line_graph(  
        samples: List[Tuple[Data, Data, torch.Tensor, torch.Tensor]]
    ):
        graphs, line_graphs, lattice, labels = map(list, zip(*samples)) 
        batched_graph = Batch.from_data_list(graphs) 
        batched_line_graph = Batch.from_data_list(line_graphs) 
        if len(labels[0].size()) > 0: 
            return batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice]), torch.stack(labels) 
        else:
            return batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice]), torch.tensor(labels) 


def canonize_edge(
    src_id, 
    dst_id,
    src_image, 
    dst_image, 
):

    if dst_id < src_id: 
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0) 

    return src_id, dst_id, src_image, dst_image 

def nearest_neighbor_edges_submit(
    atoms=None, 
    cutoff=8,
    max_neighbors=12,
    id=None, 
    use_canonize=False, 
    use_lattice=True, 
    use_angle=True, 
):

    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff) 
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors) 

    attempt = 0
    if min_nbrs < max_neighbors: 
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors): 


        neighborlist = sorted(neighborlist, key=lambda x: x[2]) 
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])


        max_dist = distances[max_neighbors - 1] 
        ids = ids[distances <= max_dist] 
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize: 
                edges[(src_id, dst_id)].add(dst_image)
            else:  
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice: 
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 1, 0])))
            
    return edges 

def pair_nearest_neighbor_edges( 
        atoms=None,
        pair_wise_distances=6, 
        use_lattice=True,
        use_angle=True,
):

    smallest = pair_wise_distances
    lattice_list = torch.as_tensor(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]).float()


    lattice = torch.as_tensor(atoms.lattice_mat).float()
    pos = torch.as_tensor(atoms.cart_coords)
    atom_num = pos.size(0)
    lat = atoms.lattice
    radius_needed = min(lat.a, lat.b, lat.c) * (smallest / 2 - 1e-9)
    r_a = (np.floor(radius_needed / lat.a) + 1).astype(np.int) 
    r_b = (np.floor(radius_needed / lat.b) + 1).astype(np.int)
    r_c = (np.floor(radius_needed / lat.c) + 1).astype(np.int)
    period_list = np.array([l for l in itertools.product(*[list(range(-r_a, r_a + 1)), list(range(-r_b, r_b + 1)), list(range(-r_c, r_c + 1))])]) 
    period_list = torch.as_tensor(period_list).float()
    n_cells = period_list.size(0)
    offset = torch.matmul(period_list, lattice).view(n_cells, 1, 3)
    expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offset).transpose(0, 1).contiguous() 
    
    dist = (pos.unsqueeze(1).unsqueeze(1) - expand_pos.unsqueeze(0))  
    dist2, index = torch.sort(dist.norm(dim=-1), dim=-1, stable=True)
    max_value = dist2[:, :, smallest - 1] 
    mask = (dist.norm(dim=-1) <= max_value.unsqueeze(-1))  

    shift = torch.matmul(lattice_list, lattice).repeat(atom_num, 1)
    shift_src = torch.arange(atom_num).unsqueeze(-1).repeat(1, lattice_list.size(0))
    shift_src = torch.cat([shift_src[i,:] for i in range(shift_src.size(0))])
    
    indices = torch.where(mask)
    dist_target = dist[indices] 
    u, v, _ = indices 

    if use_lattice:
        u = torch.cat((u, shift_src), dim=0)
        v = torch.cat((v, shift_src), dim=0)
        dist_target = torch.cat((dist_target, shift), dim=0)
        assert u.size(0) == dist_target.size(0)

    return u, v, dist_target 


def build_undirected_edgedata(
    atoms=None, 
    edges={}, 
):

    u, v, r = [], [], [] 
    for (src_id, dst_id), images in edges.items(): 

        for dst_image in images:
            
            dst_coord = atoms.frac_coords[dst_id] + dst_image 
          
            d = atoms.lattice.cart_coords( 
                dst_coord - atoms.frac_coords[src_id] 
            )

            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]: 
                u.append(uu)
                v.append(vv)
                r.append(dd)

    u = torch.tensor(u) 
    v = torch.tensor(v) 
    r = torch.tensor(r).type(torch.get_default_dtype()) 

    return u, v, r


class PygGraph(object):


    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
       
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    
    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0, 
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice: bool = True,
        use_angle: bool = True,
        four_body: bool = False,
    ):

        if neighbor_strategy == "k-nearest": 
            edges = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            u, v, r = build_undirected_edgedata(atoms, edges) 
        elif neighbor_strategy == "pairwise-k-nearest":
            u, v, r = pair_nearest_neighbor_edges(
                atoms=atoms,
                pair_wise_distances=2,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        


        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features)) 
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )

        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
        g = Data(x=node_features, edge_index=edge_index, edge_attr=r) 

        if compute_line_graph:
            linegraph_trans = LineGraph(force_directed=True) 
            g_new = Data()
            g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr 
            lg = linegraph_trans(g)

            ########################################
            lg.edge_attr = pyg_compute_linegraph_features(lg, use_four_body=four_body)  

            if four_body:
                assert lg.edge_attr.size(-1) == 3, f"expected 3-d edge_attr with four_body, got {lg.edge_attr.shape}"
            else:
                assert lg.edge_attr.size(-1) == 1, f"expected 1-d edge_attr without four_body, got {lg.edge_attr.shape}"

            ########################################


            return g_new, lg 
        else:
            return g 



##############################################################################

def _safe_norm(v, eps=1e-9):
    return torch.clamp(v.norm(dim=-1, keepdim=True), min=eps)

def compute_dihedral(b1, b2, b3):
 
    b1n = b1 / _safe_norm(b1)
    b2n = b2 / _safe_norm(b2)
    b3n = b3 / _safe_norm(b3)
    n1 = torch.cross(b1n, b2n, dim=-1)
    n2 = torch.cross(b2n, b3n, dim=-1)
    m1 = torch.cross(n1, b2n, dim=-1)
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)
    return torch.atan2(y, x)

def pyg_compute_linegraph_features(lg, use_four_body: bool = False):

    src, dst = lg.edge_index
    x = lg.x  
    r1 = -x[src]
    r2 = x[dst]
    cos_theta = (r1 * r2).sum(-1) / (_safe_norm(r1).squeeze(-1) * _safe_norm(r2).squeeze(-1))
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0).unsqueeze(-1)
    if not use_four_body or lg.num_edges == 0:
        return cos_theta

    E = lg.edge_index.size(1)
    num_nodes = x.size(0)
    buckets = [[] for _ in range(num_nodes)]
    for eid in range(E):
        buckets[int(src[eid])].append(eid)
    cos_list, sin_list = [], []
    for e12 in range(E):
        mid = int(dst[e12])
        next_edges = buckets[mid]
        if len(next_edges) == 0:
            cos_list.append(torch.tensor(0.0, device=x.device))
            sin_list.append(torch.tensor(0.0, device=x.device))
            continue
        b1 = x[src[e12]]
        b2 = x[dst[e12]]
        b3_stack = x[dst[torch.tensor(next_edges, device=x.device, dtype=torch.long)]]
        phi = compute_dihedral(b1.expand_as(b3_stack), b2.expand_as(b3_stack), b3_stack)
        cos_list.append(torch.cos(phi).mean())
        sin_list.append(torch.sin(phi).mean())
    cos_phi = torch.stack(cos_list).unsqueeze(-1)
    sin_phi = torch.stack(sin_list).unsqueeze(-1)
    return torch.cat([cos_theta, cos_phi, sin_phi], dim=-1)
##############################################################################





def pyg_compute_bond_cosines(lg):

    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


def pyg_compute_bond_angle(lg):

    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    a = (r1 * r2).sum(dim=-1) 
    b = torch.cross(r1, r2).norm(dim=-1) 
    angle = torch.atan2(b, a)
    return angle

class PygStandardize(torch.nn.Module): 

    def __init__(self, mean: torch.Tensor, std: torch.Tensor): 
        
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: Data):
       
        h = g.x 
        g.x = (h - self.mean) / self.std 
        return g


def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):

    g, t = batch 
    batch = ( 
        g.to(device), 
        t.to(device, non_blocking=non_blocking), 
    ) 

    return batch


def prepare_pyg_line_graph_batch(
    batch: Tuple[Tuple[Data, Data, torch.Tensor], torch.Tensor],
    device=None,
    non_blocking=False,
):
    
    g, lg, lattice, t = batch
    batch = (
        (
            g.to(device), 
            lg.to(device), 
            lattice.to(device, non_blocking=non_blocking), 
        ), 
        t.to(device, non_blocking=non_blocking), 
    )

    return batch 

