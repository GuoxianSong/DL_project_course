/*
sample code for reading Exp_Pca.bin and convert euler angle to rotation matrix.
use Eigen Library (http://eigen.tuxfamily.org) for vector and matrix.
*/


int point_num = 53215; // number of vertices
int dim_exp; // expression pca dim
Eigen::MatrixXf base_exp; // expression basis
Eigen::VectorXf mu_exp; // expression mean


// Read Expression Pca
void ReadExpPca()
{
	FILE* pfile = fopen('Exp_Pca.bin', 'rb');
	fread(&dim_exp, sizeof(int), 1, pfile);

	mu_exp.resize(3*point_num);
	base_exp.resize(point_num * 3, dim_exp);

    fread(mu_exp.data(), sizeof(float), point_num*3, pfile);
    fread(base_exp.data(), sizeof(float), point_num*3*dim_exp, pfile);
    fclose(pfile);
}


// Convert euler angle to rotation matrix
Eigen::MatrixXf Euler2Rot(Eigen::Vector3f euler)
{
	std::vector<Eigen::MatrixXf> Rxyz(3);
	for (int i = 0; i < Rxyz.size(); i++)
    {
        Rxyz[i].resize(3, 3);
        Rxyz[i].setZero();
        int i0 = i, i1 = (i + 1) % 3, i2 = (i + 2) % 3;
        Rxyz[i](i0, i0) = 1;
        Rxyz[i](i1, i1) = Rxyz[i](i2, i2) = cos(euler(i));
        Rxyz[i](i1, i2) = sin(euler(i));
        Rxyz[i](i2, i1) = -sin(euler(i));
    }
    return Rxyz[0] * Rxyz[1] * Rxyz[2];
}
