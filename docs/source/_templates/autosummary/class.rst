{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

	{% block methods %}
	{% if methods %}
	.. rubric:: Methods

	.. autosummary::
		:toctree: {{ objname }}
	{% for item in methods %}
		~{{ name }}.{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}
