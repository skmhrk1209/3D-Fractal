#version 150

#define SDF_DERIVATIVE_TYPE 1
#define NORMAL_DERIVATIVE_TYPE 0

// ================ definitions ================ //

// ---------------- rendering components ---------------- //

struct App
{
    float time;
    vec2 resolution;
};

struct Camera
{
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec3 position;
};

struct PointLight
{
    vec3 position;
    vec4 ambientColor;
    vec4 diffuseColor;
    vec4 specularColor;
};

struct DirectionalLight
{
    vec3 direction;
    vec4 ambientColor;
    vec4 diffuseColor;
    vec4 specularColor;
};

struct Material
{
    vec4 ambientColor;
    vec4 diffuseColor;
    vec4 specularColor;
    vec4 emissionColor;
    float shininess;
    float refractiveIndex;
};

struct March
{
    int numIterations;
    float convergenceCriteria;
    float finiteDifferenceEpsilon;
};

// ---------------- primitives ---------------- //

struct Line
{
    vec3 position;
    vec3 direction;
};

struct Sphere
{
    float radius;
    vec3 position;
};

struct Box
{
    vec3 size;
    vec3 position;
    // mat3 orientation;
};

struct Intersection
{
    vec3 position;
    bool intersected;
};

// ---------------- scene components ---------------- //

struct FractalParams
{
    int power;
    int numIterations;
    float escapeCriteria;
};

struct Fractal
{
    FractalParams params;
    Material material;
};

struct Scene
{
    vec4 backgroundColor;
    
    DirectionalLight light;
    Sphere boundingSphere;
    Fractal fractal;
};

// ================ variables ================ //

// ---------------- rendering components ---------------- //

uniform App uApp;
uniform Camera uCamera;
uniform March uMarch = March(300, 1e-4, 1e-4);

// ---------------- scene components ---------------- //

uniform Scene uScene = Scene(
    vec4(vec3(0.05), 1.0),
    DirectionalLight(vec3(-1.0, -1.0, -1.0), vec4(vec3(0.05), 1.0), vec4(vec3(0.8), 1.0), vec4(vec3(1.0), 1.0)),
    Sphere(1.7, vec3(0.0)),
    Fractal(FractalParams(2, 10, 2.0), Material(vec4(vec3(0.1), 1.0), vec4(vec3(0.8), 1.0), vec4(vec3(1.0), 1.0), vec4(vec3(0.0), 1.0), 64.0, 1.5))
);

// ---------------- varying ---------------- //

out vec4 oFragColor;

// ================ functions ================ //

// ---------------- utilities ---------------- //

vec2 linmap(vec2 in_val, vec2 in_min, vec2 in_max, vec2 out_min, vec2 out_max)
{
    return (in_val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min;
}

// ---------------- primitives ---------------- //

Intersection intersectionSphereLine(Sphere sphere, Line line)
{
    vec3 difference = line.position - sphere.position;
    float a = dot(line.direction, line.direction);
    float b = 2 * dot(difference, line.direction);
    float c = dot(difference, difference) - pow(sphere.radius, 2);
    float d = pow(b, 2) - 4 * a * c;
    float t = (-b - sqrt(d)) / (2 * a);
    return Intersection(line.position + t * line.direction, d >= 0);
}

float sdfSphere(Sphere sphere, vec3 position)
{
    return length(position - sphere.position) - sphere.radius;
}

vec3 normalSphere(Sphere sphere, vec3 position)
{
    return normalize(position - sphere.position);
}

float sdfBox(Box box, vec3 position)
{
    position = (position - box.position); // * box.orientation;
    vec3 difference = abs(position) - box.size;
    return length(max(difference, 0.0)) + min(max(difference.x, max(difference.y, difference.z)), 0.0);
}

vec3 normalBox(Box box, vec3 position, float finiteDifferenceEpsilon)
{
    return normalize(vec3(
        sdfBox(box, vec3(position.x + finiteDifferenceEpsilon, position.y, position.z)) - sdfBox(box, vec3(position.x - finiteDifferenceEpsilon, position.y, position.z)),
        sdfBox(box, vec3(position.x, position.y + finiteDifferenceEpsilon, position.z)) - sdfBox(box, vec3(position.x, position.y - finiteDifferenceEpsilon, position.z)),
        sdfBox(box, vec3(position.x, position.y, position.z + finiteDifferenceEpsilon)) - sdfBox(box, vec3(position.x, position.y, position.z - finiteDifferenceEpsilon))
    ));
}

// ---------------- complex ---------------- //

vec2 cAdd(vec2 c1, vec2 c2)
{
    // return vec2(c1.x + c2.x, c1.y + c2.y);
    return c1 + c2;
}

vec2 cSub(vec2 c1, vec2 c2)
{
    // return vec2(c1.x - c2.x, c1.y - c2.y);
    return c1 - c2;
}

vec2 cMul(vec2 c1, vec2 c2)
{
    return vec2(c1.x * c2.x - c1.y * c2.y, c1.y * c2.x + c1.x * c2.y);
}

vec2 cConj(vec2 c)
{
    return vec2(c.x, -c.y);
}

float cNorm(vec2 c)
{
    // return sqrt(cMul(c, cConj(c)).x);
    return length(c);
}

vec2 cInv(vec2 c)
{
    return cConj(c) / pow(cNorm(c), 2);
}

vec2 cDiv(vec2 c1, vec2 c2)
{
    return cMul(c1, cInv(c2));
}

vec2 cPow(vec2 c, int n)
{
    vec2 p = vec2(1.0, 0.0);
    for (int i = 0; i < n; ++i)
    {
        p = cMul(p, c);
    }
    return p;
}

// ---------------- quaternion ---------------- //

vec4 qAdd(vec4 q1, vec4 q2)
{
    // return vec4(q1.x + q2.x, q1.yzw + q2.yzw);
    return q1 + q2;
}

vec4 qSub(vec4 q1, vec4 q2)
{
    // return vec4(q1.x - q2.x, q1.yzw - q2.yzw);
    return q1 - q2;
}

vec4 qMul(vec4 q1, vec4 q2)
{
    return vec4(q1.x * q2.x - dot(q1.yzw, q2.yzw), q2.x * q1.yzw + q1.x * q2.yzw + cross(q1.yzw, q2.yzw));
}

vec4 qConj(vec4 q)
{
    return vec4(q.x, -q.yzw);
}

float qNorm(vec4 q)
{
    // return sqrt(qMul(q, qConj(q)).x);
    return length(q);
}

vec4 qInv(vec4 q)
{
    return qConj(q) / pow(qNorm(q),2);
}

vec4 qDiv(vec4 q1, vec4 q2)
{
    return qMul(q1, qInv(q2));
}

vec4 qPow(vec4 q, int n)
{
    vec4 p = vec4(1.0, vec3(0.0));
    for (int i = 0; i < n; ++i)
    {
        p = qMul(p, q);
    }
    return p;
}

// ---------------- dual ---------------- //

struct DualQ
{
    vec4 q;
    vec4 d;
};

DualQ dqAdd(DualQ dq1, DualQ dq2)
{
    return DualQ(qAdd(dq1.q, dq2.q), qAdd(dq1.d, dq2.d));
}

DualQ dqSub(DualQ dq1, DualQ dq2)
{
    return DualQ(qSub(dq1.q, dq2.q), qSub(dq1.d, dq2.d));
}

DualQ dqMul(DualQ dq1, DualQ dq2)
{
    return DualQ(qMul(dq1.q, dq2.q), qAdd(qMul(dq1.d, dq2.q), qMul(dq1.q, dq2.d)));
}

DualQ dqDiv(DualQ dq1, DualQ dq2)
{
    return DualQ(qDiv(dq1.q, dq2.q), qDiv(qSub(qMul(dq1.d, dq2.q), qMul(dq1.q, dq2.d)), qMul(dq2.q, dq2.q)));
}

DualQ dqPow(DualQ dq, int n)
{
    DualQ dp = DualQ(vec4(1.0, vec3(0.0)), vec4(0.0, vec3(0.0)));
    for (int i = 0; i < n; ++i)
    {
        dp = dqMul(dp, dq);
    }
    return dp;
}

// ---------------- fractals ---------------- //

#if SDF_DERIVATIVE_TYPE == 0
float sdfJulia(vec4 z, vec4 c, int power, int numIterations, float escapeCriteria)
{
    DualQ dzx = DualQ(z, vec4(1.0, 0.0, 0.0, 0.0));
    DualQ dcx = DualQ(c, vec4(0.0, 0.0, 0.0, 0.0));
    
    for (int i = 0; i < numIterations; ++i)
    {
        // forward-mode automatic differentiation
        dzx = dqAdd(dqPow(dzx, power), dcx);
        
        if (qNorm(dzx.q) > escapeCriteria) break;
    }
    
    return (qNorm(dzx.q) * log(qNorm(dzx.q))) / (2 * qNorm(dzx.d));
}
float sdfMandelbrot(vec4 c, vec4 z, int power, int numIterations, float escapeCriteria)
{
    DualQ dzx = DualQ(z, vec4(0.0, 0.0, 0.0, 0.0));
    DualQ dcx = DualQ(c, vec4(1.0, 0.0, 0.0, 0.0));
    
    for (int i = 0; i < numIterations; ++i)
    {
        // forward-mode automatic differentiation
        dzx = dqAdd(dqPow(dzx, power), dcx);
        
        if (qNorm(dzx.q) > escapeCriteria) break;
    }
    
    return (qNorm(dzx.q) * log(qNorm(dzx.q))) / (2 * qNorm(dzx.d));
}
#elif SDF_DERIVATIVE_TYPE == 1
float sdfJulia(vec4 z, vec4 c, int power, int numIterations, float escapeCriteria)
{
    vec4 dzx = vec4(1.0, 0.0, 0.0, 0.0);
    vec4 dcx = vec4(0.0, 0.0, 0.0, 0.0);
    
    for (int i = 0; i < numIterations; ++i)
    {
        vec4 zp = qPow(z, power - 1);
        
        // forward-mode manual differentiation
        dzx = qAdd(power * qMul(zp, dzx), dcx);
        
        z = qAdd(qMul(zp, z), c);
        
        if (qNorm(z) > escapeCriteria) break;
    }
    
    return (qNorm(z) * log(qNorm(z))) / (2 * qNorm(dzx));
}
float sdfMandelbrot(vec4 c, vec4 z, int power, int numIterations, float escapeCriteria)
{
    vec4 dzx = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 dcx = vec4(1.0, 0.0, 0.0, 0.0);
    
    for (int i = 0; i < numIterations; ++i)
    {
        vec4 zp = qPow(z, power - 1);
        
        // forward-mode manual differentiation
        dzx = qAdd(power * qMul(zp, dzx), dcx);
        
        z = qAdd(qMul(zp, z), c);
        
        if (qNorm(z) > escapeCriteria) break;
    }
    
    return (qNorm(z) * log(qNorm(z))) / (2 * qNorm(dzx));
}
#else
#endif

#if NORMAL_DERIVATIVE_TYPE == 0
vec4 normalJulia(vec4 z, vec4 c, int power, int numIterations, float escapeCriteria)
{
    DualQ dzx = DualQ(z, vec4(1.0, 0.0, 0.0, 0.0));
    DualQ dzy = DualQ(z, vec4(0.0, 1.0, 0.0, 0.0));
    DualQ dzz = DualQ(z, vec4(0.0, 0.0, 1.0, 0.0));
    DualQ dzw = DualQ(z, vec4(0.0, 0.0, 0.0, 1.0));
    
    DualQ dcx = DualQ(c, vec4(0.0, 0.0, 0.0, 0.0));
    DualQ dcy = DualQ(c, vec4(0.0, 0.0, 0.0, 0.0));
    DualQ dcz = DualQ(c, vec4(0.0, 0.0, 0.0, 0.0));
    DualQ dcw = DualQ(c, vec4(0.0, 0.0, 0.0, 0.0));
    
    for (int i = 0; i < numIterations; ++i)
    {
        // forward-mode automatic differentiation
        dzx = dqAdd(dqPow(dzx, power), dcx);
        dzy = dqAdd(dqPow(dzy, power), dcy);
        dzz = dqAdd(dqPow(dzz, power), dcz);
        dzw = dqAdd(dqPow(dzw, power), dcw);
        
        if (qNorm(dzx.q) > escapeCriteria) break;
    }
    
    mat4 J = mat4(dzx.d, dzy.d, dzz.d, dzw.d);
    return dzx.q * J;
}
vec4 normalMandelbrot(vec4 c, vec4 z, int power, int numIterations, float escapeCriteria)
{
    DualQ dzx = DualQ(z, vec4(0.0, 0.0, 0.0, 0.0));
    DualQ dzy = DualQ(z, vec4(0.0, 0.0, 0.0, 0.0));
    DualQ dzz = DualQ(z, vec4(0.0, 0.0, 0.0, 0.0));
    DualQ dzw = DualQ(z, vec4(0.0, 0.0, 0.0, 0.0));
    
    DualQ dcx = DualQ(c, vec4(1.0, 0.0, 0.0, 0.0));
    DualQ dcy = DualQ(c, vec4(0.0, 1.0, 0.0, 0.0));
    DualQ dcz = DualQ(c, vec4(0.0, 0.0, 1.0, 0.0));
    DualQ dcw = DualQ(c, vec4(0.0, 0.0, 0.0, 1.0));
    
    for (int i = 0; i < numIterations; ++i)
    {
        // forward-mode automatic differentiation
        dzx = dqAdd(dqPow(dzx, power), dcx);
        dzy = dqAdd(dqPow(dzy, power), dcy);
        dzz = dqAdd(dqPow(dzz, power), dcz);
        dzw = dqAdd(dqPow(dzw, power), dcw);
        
        if (qNorm(dzx.q) > escapeCriteria) break;
    }
    
    mat4 J = mat4(dzx.d, dzy.d, dzz.d, dzw.d);
    return dzx.q * J;
}
#elif NORMAL_DERIVATIVE_TYPE == 1
vec4 normalJulia(vec4 z, vec4 c, int power, int numIterations, float escapeCriteria)
{
    vec4 dzx = vec4(1.0, 0.0, 0.0, 0.0);
    vec4 dzy = vec4(0.0, 1.0, 0.0, 0.0);
    vec4 dzz = vec4(0.0, 0.0, 1.0, 0.0);
    vec4 dzw = vec4(0.0, 0.0, 0.0, 1.0);
    
    vec4 dcx = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 dcy = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 dcz = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 dcw = vec4(0.0, 0.0, 0.0, 0.0);
    
    for (int i = 0; i < numIterations; ++i)
    {
        vec4 zp = qPow(z, power - 1);
        
        // forward-mode manual differentiation
        dzx = qAdd(power * qMul(zp, dzx), dcx);
        dzy = qAdd(power * qMul(zp, dzy), dcy);
        dzz = qAdd(power * qMul(zp, dzz), dcz);
        dzw = qAdd(power * qMul(zp, dzw), dcw);
        
        z = qAdd(qMul(zp, z), c);
        
        if (qNorm(z) > escapeCriteria) break;
    }
    
    mat4 J = mat4(dzx, dzy, dzz, dzw);
    return z * J;
}
vec4 normalMandelbrot(vec4 c, vec4 z, int power, int numIterations, float escapeCriteria)
{
    vec4 dzx = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 dzy = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 dzz = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 dzw = vec4(0.0, 0.0, 0.0, 0.0);
    
    vec4 dcx = vec4(1.0, 0.0, 0.0, 0.0);
    vec4 dcy = vec4(0.0, 1.0, 0.0, 0.0);
    vec4 dcz = vec4(0.0, 0.0, 1.0, 0.0);
    vec4 dcw = vec4(0.0, 0.0, 0.0, 1.0);
    
    for (int i = 0; i < numIterations; ++i)
    {
        vec4 zp = qPow(z, power - 1);
        
        // forward-mode manual differentiation
        dzx = qAdd(power * qMul(zp, dzx), dcx);
        dzy = qAdd(power * qMul(zp, dzy), dcy);
        dzz = qAdd(power * qMul(zp, dzz), dcz);
        dzw = qAdd(power * qMul(zp, dzw), dcw);
        
        z = qAdd(qMul(zp, z), c);
        
        if (qNorm(z) > escapeCriteria) break;
    }
    
    mat4 J = mat4(dzx, dzy, dzz, dzw);
    return z * J;
}
#else
vec4 normalJulia(vec4 z, vec4 c, int numIterations, float escapeCriteria)
{
    // jacobian
    mat4 Jz = mat4(1.0);
    mat4 Jc = mat4(0.0);
    
    for (int i = 0; i < numIterations; ++i)
    {
        // forward-mode automatic differentiation
        // NOTE: glsl uses column-major matrices
        Jz = 2.0 * mat4(
            +z.x, z.y, z.z, z.w,
            -z.y, z.x, 0.0, 0.0,
            -z.z, 0.0, z.x, 0.0,
            -z.w, 0.0, 0.0, z.x
        ) * Jz + Jc;
        
        z = qAdd(qPow(z, 2), c);
        
        if (qNorm(z) > escapeCriteria) break;
    }
    
    return z * Jz;
}
vec4 normalMandelbrot(vec4 c, vec4 z, int numIterations, float escapeCriteria)
{
    // jacobian
    mat4 Jz = mat4(0.0);
    mat4 Jc = mat4(1.0);
    
    for (int i = 0; i < numIterations; ++i)
    {
        // forward-mode automatic differentiation
        // NOTE: glsl uses column-major matrices
        Jz = 2.0 * mat4(
            +z.x, z.y, z.z, z.w,
            -z.y, z.x, 0.0, 0.0,
            -z.z, 0.0, z.x, 0.0,
            -z.w, 0.0, 0.0, z.x
        ) * Jz + Jc;
        
        z = qAdd(qPow(z, 2), c);
        
        if (qNorm(z) > escapeCriteria) break;
    }
    
    return z * Jz;
}
#endif

// ---------------- reflection ---------------- //

vec4 phongReflection(vec3 surfaceNormal, vec3 eyeDirection, DirectionalLight light, Material material)
{
    surfaceNormal = normalize(surfaceNormal);
    eyeDirection = normalize(eyeDirection);
    light.direction = normalize(light.direction);
    
    vec4 ambientColor = light.ambientColor * material.ambientColor;
    vec4 diffuseColor = light.diffuseColor * material.diffuseColor;
    vec4 specularColor = light.specularColor * material.specularColor;
    
    diffuseColor *= max(dot(-light.direction, surfaceNormal), 0.0);
    specularColor *= pow(max(dot(reflect(light.direction, surfaceNormal), -eyeDirection), 0.0), material.shininess);
    
    vec4 color = clamp(ambientColor + diffuseColor + specularColor + material.emissionColor, 0.0, 1.0);
    return color;
}

float fresnelReflectance(vec3 light, vec3 normal, float refractiveIndex)
{
    float specularReflectance = pow((1.0 - refractiveIndex) / (1.0 + refractiveIndex), 2.0);
    return specularReflectance + (1.0 - specularReflectance) * pow(1.0 - dot(-light, normal), 5.0);
}

// ---------------- noise ---------------- //

//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
//

float mod289(float x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 mod289(vec3 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float permute(float x)
{
    return mod289(((x * 34.0) + 1.0) * x);
}

vec2 permute(vec2 x)
{
    return mod289(((x * 34.0) + 1.0) * x);
}

vec3 permute(vec3 x)
{
    return mod289(((x * 34.0) + 1.0) * x);
}

vec4 permute(vec4 x)
{
    return mod289(((x * 34.0) + 1.0) * x);
}

float taylorInvSqrt(float r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

vec2 taylorInvSqrt(vec2 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 taylorInvSqrt(vec3 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 grad4(float j, vec4 ip)
{
    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
    vec4 p, s;

    p.xyz = floor(fract(vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    s = vec4(lessThan(p, vec4(0.0)));
    p.xyz = p.xyz + (s.xyz * 2.0 - 1.0) * s.www;

    return p;
}

// (sqrt(5) - 1)/4 = F4, used once below
#define F4 0.309016994374947451

float snoise(vec2 v)
{
    const vec4 C = vec4(0.211324865405187,    // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,    // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626, // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
                                            // First corner
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);

    // Other corners
    vec2 i1;
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));

    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    // Compute final noise value at P
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float snoise(vec3 v)
{
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;       // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i);
    vec4 p = permute(permute(permute(
                                 i.z + vec4(0.0, i1.z, i2.z, 1.0)) +
                             i.y + vec4(0.0, i1.y, i2.y, 1.0)) +
                     i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z); //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_); // mod(j,N)

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 105.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1),
                                   dot(p2, x2), dot(p3, x3)));
}

float snoise(vec4 v)
{
    const vec4 C = vec4(0.138196601125011,     // (5 - sqrt(5))/20  G4
                        0.276393202250021,     // 2 * G4
                        0.414589803375032,     // 3 * G4
                        -0.447213595499958); // -1 + 4 * G4

    // First corner
    vec4 i = floor(v + dot(v, vec4(F4)));
    vec4 x0 = v - i + dot(i, C.xxxx);

    // Other corners

    // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
    vec4 i0;
    vec3 isX = step(x0.yzw, x0.xxx);
    vec3 isYZ = step(x0.zww, x0.yyz);
    //  i0.x = dot( isX, vec3( 1.0 ) );
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    //  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;

    // i0 now contains the unique values 0,1,2,3 in each channel
    vec4 i3 = clamp(i0, 0.0, 1.0);
    vec4 i2 = clamp(i0 - 1.0, 0.0, 1.0);
    vec4 i1 = clamp(i0 - 2.0, 0.0, 1.0);

    //  x0 = x0 - 0.0 + 0.0 * C.xxxx
    //  x1 = x0 - i1  + 1.0 * C.xxxx
    //  x2 = x0 - i2  + 2.0 * C.xxxx
    //  x3 = x0 - i3  + 3.0 * C.xxxx
    //  x4 = x0 - 1.0 + 4.0 * C.xxxx
    vec4 x1 = x0 - i1 + C.xxxx;
    vec4 x2 = x0 - i2 + C.yyyy;
    vec4 x3 = x0 - i3 + C.zzzz;
    vec4 x4 = x0 + C.wwww;

    // Permutations
    i = mod289(i);
    float j0 = permute(permute(permute(permute(i.w) + i.z) + i.y) + i.x);
    vec4 j1 = permute(permute(permute(permute(
                                          i.w + vec4(i1.w, i2.w, i3.w, 1.0)) +
                                      i.z + vec4(i1.z, i2.z, i3.z, 1.0)) +
                              i.y + vec4(i1.y, i2.y, i3.y, 1.0)) +
                      i.x + vec4(i1.x, i2.x, i3.x, 1.0));

    // Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
    // 7*7*6 = 294, which is close to the ring size 17*17 = 289.
    vec4 ip = vec4(1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0);

    vec4 p0 = grad4(j0, ip);
    vec4 p1 = grad4(j1.x, ip);
    vec4 p2 = grad4(j1.y, ip);
    vec4 p3 = grad4(j1.z, ip);
    vec4 p4 = grad4(j1.w, ip);

    // Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    p4 *= taylorInvSqrt(dot(p4, p4));

    // Mix contributions from the five corners
    vec3 m0 = max(0.6 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0);
    vec2 m1 = max(0.6 - vec2(dot(x3, x3), dot(x4, x4)), 0.0);
    m0 = m0 * m0;
    m1 = m1 * m1;
    return 49.0 * (dot(m0 * m0, vec3(dot(p0, x0), dot(p1, x1), dot(p2, x2))) + dot(m1 * m1, vec2(dot(p3, x3), dot(p4, x4))));
}

// ---------------- ray marching ---------------- //

vec4 rayMarching(App app, Scene scene, March march, Line ray)
{
    Intersection intersection = intersectionSphereLine(scene.boundingSphere, ray);
    
    if (intersection.intersected)
    {
        ray.position = intersection.position;
        
        vec4 juliaType = 0.45 * cos(vec4(0.5, 3.9, 1.4, 1.1) + app.time * 0.15 * vec4(1.2, 1.7, 1.3, 2.5)) - vec4(0.3, 0.0, 0.0, 0.0);
        vec4 criticalPoint = vec4(0.0);
        
        for (int i = 0; i < march.numIterations; ++i)
        {
            float sdf = sdfJulia(vec4(ray.position, 0.0), juliaType, scene.fractal.params.power, scene.fractal.params.numIterations, scene.fractal.params.escapeCriteria);
            
            // ray marching
            ray.position += sdf * ray.direction;
            
            // collision detection
            if (abs(sdf) < march.convergenceCriteria)
            {
                vec3 surfaceNormal = normalize(normalJulia(vec4(ray.position, 0.0), juliaType, scene.fractal.params.power, scene.fractal.params.numIterations, scene.fractal.params.escapeCriteria).xyz);
                
                vec4 fragColor = phongReflection(surfaceNormal, ray.direction, scene.light, scene.fractal.material);
                
                return fragColor;
            }
            
            if (sdfSphere(scene.boundingSphere, ray.position) > 0) break;
        }
    }
    
    return scene.backgroundColor;
}

void main()
{
    vec2 fragCoord = linmap(gl_FragCoord.xy, vec2(0, 0), uApp.resolution, vec2(-1, -1), vec2(1, 1));
    
    // why this does not work?
    // vec3 rayDirection = normalize((inverse(uViewMatrix) * inverse(uProjectionMatrix) * vec4(vec3(fragCoord, 1.0), 0.0)).xyz);
    vec3 rayDirection = normalize((inverse(mat3(uCamera.viewMatrix)) * inverse(mat3(uCamera.projectionMatrix)) * vec3(fragCoord, 1.0)).xyz);
    
    Line ray = Line(uCamera.position, rayDirection);
    oFragColor = rayMarching(uApp, uScene, uMarch, ray);
}
