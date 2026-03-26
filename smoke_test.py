import sys
sys.path.insert(0, '.')

print("Loading models...")
from backend.scheduler import GNNScheduler, SAGEGNNScheduler

gat  = GNNScheduler(model_path='model/scheduler_model.pt')
sage = SAGEGNNScheduler(model_path='model/scheduler_model_sage.pt')
print("Both models loaded.\n")

machines = [
    {
        'machine_id': 'machine-%03d' % i,
        'total_cpu': 4.0 + i,
        'total_ram': 16.0 + i * 2,
        'available_cpu': 3.0 + i * 0.5,
        'available_ram': 12.0 + i,
        'load': 0.1 + i * 0.03,
        'bandwidth': 2.0 + i * 0.3,
    }
    for i in range(20)
]

task = {'cpu_request': 2.0, 'memory_request': 4.0, 'priority': 5}

gat_idx  = gat.predict(task, machines)
sage_idx = sage.predict(task, machines)

print(f"GAT   chose: {machines[gat_idx]['machine_id']}")
print(f"SAGE  chose: {machines[sage_idx]['machine_id']}")
print("\n[OK] Both models load and predict correctly")
