"""Synthetic resolved-node families for semantic adapter integration tests."""

from docutils import nodes
from docutils.parsers.rst import Directive


class DropdownNode(nodes.container):
    pass


class TabNode(nodes.container):
    pass


class YoutubeNode(nodes.General, nodes.Element):
    pass


class UnknownSemanticLeaf(nodes.General, nodes.Element):
    def astext(self):
        return str(self.get("text", ""))


class UnknownStructuralBox(nodes.container):
    pass


class _BodyDirective(Directive):
    has_content = True

    node_class = nodes.container

    def run(self):
        node = self.node_class()
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class DropdownDirective(_BodyDirective):
    node_class = DropdownNode

    def run(self):
        result = super().run()
        result[0]["title"] = "All dropdown content"
        return result


class TabDirective(_BodyDirective):
    node_class = TabNode

    def run(self):
        result = super().run()
        result[0]["label"] = "Python"
        return result


class UnknownBoxDirective(_BodyDirective):
    node_class = UnknownStructuralBox


class YoutubeDirective(Directive):
    required_arguments = 1
    optional_arguments = 1
    final_argument_whitespace = True

    def run(self):
        node = YoutubeNode()
        node["url"] = self.arguments[0]
        node["title"] = self.arguments[1] if len(self.arguments) > 1 else "Video"
        node["provider"] = "YouTube"
        return [node]


class UnknownLeafDirective(Directive):
    required_arguments = 1
    final_argument_whitespace = True

    def run(self):
        node = UnknownSemanticLeaf()
        node["text"] = self.arguments[0]
        return [node]


def _visit_passthrough(self, node):
    return None


def _depart_passthrough(self, node):
    return None


def _visit_leaf(self, node):
    self.body.append(self.encode(node.astext()))
    raise nodes.SkipNode


def setup(app):
    passthrough = (_visit_passthrough, _depart_passthrough)
    app.add_node(DropdownNode, html=passthrough)
    app.add_node(TabNode, html=passthrough)
    app.add_node(UnknownStructuralBox, html=passthrough)
    app.add_node(YoutubeNode, html=(_visit_leaf, _depart_passthrough))
    app.add_node(UnknownSemanticLeaf, html=(_visit_leaf, _depart_passthrough))
    app.add_directive("fixture-dropdown", DropdownDirective)
    app.add_directive("fixture-tab", TabDirective)
    app.add_directive("fixture-youtube", YoutubeDirective)
    app.add_directive("fixture-unknown-leaf", UnknownLeafDirective)
    app.add_directive("fixture-unknown-box", UnknownBoxDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
