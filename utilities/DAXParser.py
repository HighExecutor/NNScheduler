import xml.etree.ElementTree as ET

from env.entities import Task
from env.entities import File
from env.entities import Workflow


def read_workflow(dax_filepath, wf_name):
    parser = DAXParser()
    wf = parser.parseXml(dax_filepath, wf_name)
    return wf


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

    def parseXml(self, filepath, wf_name):
        tree = ET.parse(filepath)
        root = tree.getroot()
        jobs = root.findall('./{http://pegasus.isi.edu/schema/DAX}job')
        children = root.findall('./{http://pegasus.isi.edu/schema/DAX}child')
        id2Task = dict()
        for job in jobs:
            ## build task
            id = job.attrib['id']
            task = Task(id)
            task.runtime = float(job.attrib['runtime'])
            self.readFiles(job, task)
            id2Task[task.id] = task

        for child in children:
            id = child.attrib['ref']
            parents = [id2Task[prt.attrib['ref']] for prt in
                       child.findall('./{http://pegasus.isi.edu/schema/DAX}parent')]
            child = id2Task[id]
            child.parents.update(parents)
            for parent in parents:
                parent.children.add(child)

        heads = [task for (name, task) in id2Task.items() if len(task.parents) == 0]

        common_head = Task("000", is_head=True)
        for head in heads:
            head.parents = set([common_head])
        common_head.children = heads

        wf = Workflow(wf_name, common_head)
        return wf
