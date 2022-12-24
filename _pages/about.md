---
layout: page
title: About me
image: '/images/pages/donghuna_profile.jpeg'
---

---

###### <center>Korea University of Technology and Education - Computer Science (2005 - 2013)</center>
###### <center>Samsung Software Membership (2012 - 2013)</center>

###### <center>Samsung Electronics Visual Display (2013 - 2017)</center>
###### <center>Samsung Advanced Institute of Technology (2017 - 2021)</center>
###### <center>Samsung Electronics Innovation Center (2021 - Current)</center>

---


###### <center>Daejeon Local Skills Competition - Silver medal (2004) </center>
###### <center>The 39th National Skills Competition - Bronze medal (2004)</center>
###### <center>ACM-ICPC #88 (2009)</center>
###### <center>Audio product Front Micom & SCM (2013 - 2017)</center>
###### <center>Fingerprint Recognition Algorithm (2017 - 2020)</center>
###### <center>Face Recognition Deep Learning (2020 - 2021)</center>
###### <center>Software Engineering (2021 - Current)</center>
###### <center>Samsung Software Certification - Professional</center>
###### <center>Associate Architect</center>
###### <center>Best Code Reviewer</center>

<!-- Include the library. -->
<script src="https://unpkg.com/github-calendar@latest/dist/github-calendar.min.js"></script>

<!-- Optionally, include the theme (if you don't want to struggle to write the CSS) -->
<link rel="stylesheet" href="https://unpkg.com/github-calendar@latest/dist/github-calendar-responsive.css"/>

<div class="calendar1">
</div>


<script>
    GitHubCalendar(".calendar1", "donghuna", { responsive: true, tooltips: false, global_stats: false}).then(function() {
        // delete the space underneath the module bar which is caused by minheight 
        document.getElementsByClassName('calendar1')[0].style.minHeight = "100px";
        // hide more and less legen below the contribution graph
        //document.getElementsByClassName('contrib-legend')[0].style.display = "none";
    });
</script>

<br><br><br>

<!-- Prepare a container for your calendar. -->
<script src="https://cdn.rawgit.com/IonicaBizau/github-calendar/gh-pages/dist/github-calendar.min.js"></script>
<!-- Optionally, include the theme (if you don't want to struggle to write the CSS) -->
<link rel="stylesheet" href="https://cdn.rawgit.com/IonicaBizau/github-calendar/gh-pages/dist/github-calendar.css" />

<!-- Prepare a container for your calendar. -->
<div class="calendar">
</div>

<script>
    new GitHubCalendar(".calendar", "donghuna");
</script>