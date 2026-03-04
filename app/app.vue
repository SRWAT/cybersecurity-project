<template>
  <div>
    <nav class="navbar">
      <span class="logo">TruthLens</span>

      <ul class="nav-links">
        <li><a href="#detect">Detect</a></li>
        <li><a href="#how">How it works</a></li>
      </ul>
    </nav>

    <NuxtPage />
  </div>
</template>

<style>
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 48px;
  background: #080b10;
  border-bottom: 1px solid #1e2d3d;
}

.logo {
  color: white;
  font-size: 1.3rem;
  font-weight: bold;
}

.nav-links {
  display: flex;
  gap: 32px;
  list-style: none;
}

.nav-links a {
  color: #5a7a94;
  text-decoration: none;
}

.nav-links a:hover {
  color: #00e5ff;
}
</style>