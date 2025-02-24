class WaterSimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        
        
        // Particle properties
        this.dt = 0.005;  // Time step
        this.capillaryStrength = 0.01;  // Controls capillary force
        this.meanReversalTime = 1.0; // Controls reversal time
        
//        const rect = this.canvas.getBoundingClientRect();
                
        
        // Update API endpoint to be more specific
        const protocol = window.location.protocol;
        const host = window.location.host;
        this.apiUrl = `${protocol}//${host}/simulate_step`;
        
        // Initialize error message element
        this.errorMessage = document.getElementById('error-message');
        
        // Add controls
        this.capillaryStrengthInput = document.getElementById('capillaryStrength');
        this.meanReversalTimeInput = document.getElementById('meanReversalTime');
        this.capillaryStrengthValue = document.getElementById('capillaryStrengthValue');
        this.meanReversalTimeValue = document.getElementById('meanReversalTimeValue');
        
        // Add event listeners for controls
        this.capillaryStrengthInput.addEventListener('input', (e) => {
            this.capillaryStrength = parseFloat(e.target.value);
            this.capillaryStrengthValue.textContent = this.capillaryStrength.toFixed(3);
        });
        
        this.meanReversalTimeInput.addEventListener('input', (e) => {
            this.meanReversalTime = parseFloat(e.target.value);
            this.meanReversalTimeValue.textContent = this.meanReversalTime.toFixed(2);
        });
        
        // Initialize particles array
        this.particles = [];
        this.numParticles = 20; // Start with 3 particles
        
        // Initialize water heights
        this.water150 = [];
    }
    
    // Update particle position
    async update() {
        try {
            const url = `${this.apiUrl}?capillaryStrength=${this.capillaryStrength}&meanReversalTime=${this.meanReversalTime}`;
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            this.particles = data.particles;
            this.water25 = data.water25;
            this.water50 = data.water50;
            this.water100 = data.water100;
            this.water150 = data.water150;
            
            this.errorMessage.style.display = 'none';
        } catch (error) {
            console.error('Error fetching simulation data:', error);
            this.errorMessage.textContent = `Cannot connect to simulation backend: ${error.message}`;
            this.errorMessage.style.display = 'block';
        }
    }
    
    // Draw the current state
    draw() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);
    
        // Draw all particles
        for (const particle of this.particles) {
            this.drawParticle(particle.x, particle.y, particle.l, particle.r, particle.th, particle.sgn);
            this.drawParticle(particle.x, particle.y + this.height, particle.l, particle.r, particle.th, particle.sgn);
            this.drawParticle(particle.x, particle.y - this.height, particle.l, particle.r, particle.th, particle.sgn);
            this.drawParticle(particle.x + this.width, particle.y, particle.l, particle.r, particle.th, particle.sgn);
            this.drawParticle(particle.x + this.width, particle.y + this.height, particle.l, particle.r, particle.th, particle.sgn);
            this.drawParticle(particle.x + this.width, particle.y - this.height, particle.l, particle.r, particle.th, particle.sgn);
            this.drawParticle(particle.x - this.width, particle.y, particle.l, particle.r, particle.th, particle.sgn);
            this.drawParticle(particle.x - this.width, particle.y + this.height, particle.l, particle.r, particle.th, particle.sgn);
            this.drawParticle(particle.x - this.width, particle.y - this.height, particle.l, particle.r, particle.th, particle.sgn);
        }
        
        // Draw all water contours
        for (const contour of this.water25){
            this.drawContour(contour.x, contour.y, "#414584")
        }
        for (const contour of this.water50){
            this.drawContour(contour.x, contour.y, "#43848c")
        }
        for (const contour of this.water100){
            this.drawContour(contour.x, contour.y, "#7fc96c")
        }
        for (const contour of this.water150){
            this.drawContour(contour.x, contour.y, "#cadf4f")
        }
        
    }
    
    // Helper function for drawing particles
//    drawParticle(x, y){
//        this.ctx.beginPath();
//        this.ctx.arc(x, y, 10, 0, Math.PI * 2);
//        this.ctx.fillStyle = "black";
//        this.ctx.fill();
//        this.ctx.closePath();
//    }
    
    drawParticle(x, y, l, r, theta, sgn){
        const ctx = this.ctx;
        const arrowAngle = Math.PI / 6;
        const headLength = 8;
        
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(theta);
        
        ctx.beginPath();
        ctx.rect(-l, -r, 2 * l, 2 * r);
        ctx.fillStyle = "#797979";
        ctx.fill();
        ctx.closePath();
        
        ctx.beginPath();
        ctx.arc(-l , 0, r, Math.PI / 2, 3 * Math.PI / 2);
        ctx.arc(l , 0, r, -Math.PI / 2, Math.PI / 2);
        ctx.fill();
        ctx.closePath();
        
        ctx.moveTo(-l/2, 0);
        ctx.lineTo( l/2, 0);
        ctx.moveTo( sgn*l/2, 0);
        ctx.lineTo( sgn*(l/2 - headLength * Math.cos(arrowAngle)),
                    -headLength * Math.sin(arrowAngle));
        ctx.moveTo( sgn*l/2, 0);
        ctx.lineTo( sgn*(l/2 - headLength * Math.cos(arrowAngle)),
                    headLength * Math.sin(arrowAngle));
        
        ctx.strokeStyle = "white"; // Arrow color
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.restore();
    }
    
    drawArrow(ctx, x, y, al) {
        const arrowAngle = Math.PI / 6; // 30-degree angle for arrowhead
        const headLength = 5; // Length of arrowhead sides

        ctx.beginPath();
        
        // Arrow main line (extending forward)
        ctx.moveTo(x, y);
        ctx.lineTo(x + al, y);
        
        // Left arrowhead
        ctx.moveTo(x + arrowLength, y);
        ctx.lineTo(x + arrowLength - headLength * Math.cos(arrowAngle),
                   y - headLength * Math.sin(arrowAngle));
        
        // Right arrowhead
        ctx.moveTo(x + arrowLength, y);
        ctx.lineTo(x + arrowLength - headLength * Math.cos(arrowAngle),
                   y + headLength * Math.sin(arrowAngle));

        ctx.strokeStyle = "red"; // Arrow color
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    
    drawContour(xArray, yArray, color) {
        const ctx = this.ctx;
        
        if (xArray.length !== yArray.length || xArray.length === 0) {
            console.error("Invalid contour arrays: lengths do not match or are empty.");
            return;
        }

        ctx.beginPath();
        ctx.moveTo(xArray[0], yArray[0]); // Start at the first point

        for (let i = 1; i < xArray.length; i++) {
            ctx.lineTo(xArray[i], yArray[i]); // Draw line to next point
        }

        ctx.strokeStyle = color; // Contour color
        ctx.lineWidth = 2; // Line thickness
        ctx.stroke();
    }
    
    
    // Modify animate to handle async update
    async animate() {
        await this.update();
        this.draw();
        // Add a small delay to prevent overwhelming the server
        setTimeout(() => {
            requestAnimationFrame(() => this.animate());
        }, 50); // 50ms delay between updates
    }
}

// Start simulation when page loads
window.onload = () => {
    const canvas = document.getElementById('simulation');
    const simulation = new WaterSimulation(canvas);
    simulation.animate();
};
