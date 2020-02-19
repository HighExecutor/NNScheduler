import xml.etree.ElementTree as ET

from heft_deps.resource_manager import Task
from heft_deps.resource_manager import File
from heft_deps.resource_manager import AbstractWorkflow, Range


class DAXParser:
    def __init__(self):
        pass

    def readFiles(self, job, task):
        files = job.findall('./{http://pegasus.isi.edu/schema/DAX}uses')

        def buildFile(file):
            return File(file.attrib['file'], int(file.attrib['size']))
        output_files = {fl.name: fl for fl in [buildFile(file) for file in files if file.attrib['link'] == "output"]}
        input_files = {fl.name: fl for fl in [buildFile(file) for file in files if file.attrib['link'] == "input"]}
        task.output_files = output_files
        task.input_files = input_files

    def parseXml(self, filepath, wfId, taskPostfixId, wf_name, is_head=True):
        tree = ET.parse(filepath)
        root = tree.getroot()
        jobs = root.findall('./{http://pegasus.isi.edu/schema/DAX}job')
        children = root.findall('./{http://pegasus.isi.edu/schema/DAX}child')
        alternates = root.findall('./{http://pegasus.isi.edu/schema/DAX}subwf')
        subwfs = set()
        internal_id2Task = dict()

        for subwf in alternates:
            subwf_id = subwf.attrib['id']
            subwf_jobs = subwf.findall('./{http://pegasus.isi.edu/schema/DAX}job')
            subwf_tasks = dict()
            for job in subwf_jobs:
                internal_id = job.attrib['id']
                id = internal_id  # + "_" + taskPostfixId + "_" + subwf_id
                name = job.attrib['name']
                task = Task(id,internal_id,subtask=True)
                if 'range' in job.attrib:
                    range_string = job.attrib['range']
                    if range_string != None:
                        parts = range_string.split('-')
                        task.range = Range(int(parts[0]), int(parts[1]))
                if 'alternate' in job.attrib:
                    alternates_string = job.attrib['alternate']
                    if alternates_string != None:
                        parts = alternates_string.split(',')
                        task.alternate_ids = parts
                task.soft_reqs.add(name)
                task.runtime = float(job.attrib['runtime'])
                self.readFiles(job, task)
                internal_id2Task[task.id] = task
                subwf_tasks[task.id] = task

            for child in children:
                id = child.attrib['ref']
                try:
                    parents = [subwf_tasks[prt.attrib['ref']] for prt in child.findall('./{http://pegasus.isi.edu/schema/DAX}parent')]
                    child = subwf_tasks[id]
                    child.parents.update(parents)
                    for parent in parents:
                        parent.children.add(child)
                except:
                    pass

            heads = [task for (name, task) in subwf_tasks.items() if len(task.parents) == 0]
            common_head = Task("000_" + subwf_id, "000")
            common_head.runtime = 0
            for head in heads:
                head.parents = set([common_head])
            common_head.children = heads
            subwf = AbstractWorkflow(subwf_id, subwf_id, common_head)
            subwfs.add(subwf)

        for job in jobs:

            # build task

            internal_id = job.attrib['id']
            id = internal_id + "_" + taskPostfixId + "_" + wf_name
            soft = job.attrib['name']
            task = Task(id, internal_id)
            if 'range' in job.attrib:
                range_string = job.attrib['range']
                if range_string != None:
                    parts = range_string.split('-')
                    task.range = Range(int(parts[0]), int(parts[1]))
            if 'alternate' in job.attrib:
                alternates_string = job.attrib['alternate']
                if alternates_string != None:
                    parts = alternates_string.split(',')
                    task.alternate_ids = parts
            task.soft_reqs.add(soft)
            task.runtime = float(job.attrib['runtime'])
            self.readFiles(job, task)
            internal_id2Task[task.internal_wf_id] = task

        for id in internal_id2Task:
            if hasattr(internal_id2Task[id], 'alternate_ids'):
                internal_id2Task[id].alternates = []
                for alternate_id in internal_id2Task[id].alternate_ids:
                    alternate = next(subwf for subwf in subwfs if subwf.id==alternate_id)
                    internal_id2Task[id].alternates.append(alternate)

        for child in children:
            id = child.attrib['ref']
            parents = [internal_id2Task[prt.attrib['ref']] for prt in child.findall('./{http://pegasus.isi.edu/schema/DAX}parent')]
            child = internal_id2Task[id]
            child.parents.update(parents)
            for parent in parents:
                parent.children.add(child)

        heads = [task for (name, task) in internal_id2Task.items() if len(task.parents) == 0 and task.subtask == False ]

        common_head = Task("000_" + taskPostfixId, "000", is_head)
        if is_head != True:
            common_head.runtime = 0
        for head in heads:
            head.parents = set([common_head])
        common_head.children = heads

        wf = AbstractWorkflow(wfId, wf_name, common_head)
        wf.get_real_wf()
        return wf









