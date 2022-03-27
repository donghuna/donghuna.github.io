---
layout: page
title: test
image: '/images/pages/styleguide.jpg'
---

<div class='o-wrapper'>
  <div class='o-grid'>
    {% for post in paginator.posts %}
      {% include post-card.liquid %}
    {% endfor %}
  </div>
</div>
