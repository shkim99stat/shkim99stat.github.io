---
title: projects
icon: fas fa-flask
order: 4
permalink: /projects/
---

구현과 재현 실험을 모으는 공간입니다. 각 프로젝트에는 코드 저장소, 실행 조건, 결과, 한계를 함께 기록합니다.

{% assign projects = site.posts | where_exp: "post", "post.categories contains '프로젝트'" %}
{% if projects.size > 0 %}
{% for post in projects %}
### [{{ post.title }}]({{ post.url | relative_url }})
{{ post.description }}
{% endfor %}
{% else %}
아직 공개된 프로젝트 글이 없습니다.
{% endif %}
