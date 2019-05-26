#include<stdio.h>
#include<stdlib.h>
#include<dos.h>
#include<conio.h>
#include<math.h>

#define PATTERN 8
#define LROWS 5
#define LCOLS 4
#define KROWS 2
#define KCOLS 5
#define ran() ((rand() % 10000) / 10000.0 / 5) - 0.1

float x[PATTERN][LCOLS] = {{0.,0.,0.,-1},{0.,0.,1.,-1},{0.,1.,0.,-1},
                          {0.,1.,1.,-1},{1.,0.,0.,-1},{1.,0.,1.,-1},
						  {1.,1.,0.,-1},{1.,1.,1.,-1}};
float d[PATTERN][KROWS] = {{0.,0.},{0.,1.},{0.,1.},{1.,0.},
						{0.,1.},{1.,0.},{1.,0.},{1.,1.}};
						
float w_l[LROWS][LCOLS], w_k[KROWS][KCOLS];
float NET_l[LROWS], z[LROWS];

float NET_k[KROWS], OUT[KROWS];
float delta_OUT[KROWS], delta_z[LROWS];
char c;
float n = 0.;
float N = 0.;
float EMIN = 0.;
float wx = 0., wz = 0., charge = 0., delta_w = 0., E = 0.;
int i = 0, j = 0, k = 0, l = 0, m = 0;
int number = 0;

void Initialize_weight();
void Print_Weight();
void Forward_Pass();
void Backward_Pass();
void Delta_Rule();
void Amend_Weight();
void Input_x();

int main(void)
{
	do{
		printf(" Welcome to Backpropagation !");
		printf(" --- Perform Full Adder. --- ");
		printf("\n Input E_min ; ");
		scanf("%f", &EMIN );
		printf(" Input n(learning ratio) :");
		scanf("%f",&n);
		printf(" Input N(nanda) : ");
		scanf("%f",&N);
		printf("Input Maximum learning number : ");
		scanf("%d",&number);
		printf("perform Backpropagation [y/n]");
	}while((c = getch()) != 'y');
	
	Initialize_weight();
	for(l = 0; l<number; l++)
	{
		for(k = 0; k<PATTERN; k++){
			Forward_Pass();
			Backward_Pass();
			Delta_Rule();
			Amend_Weight();
		}
		if(E < EMIN) break;
		E = 0.;
	}
	Input_x();
}


void Forward_Pass()
{
	for(i = 0; i<LROWS; i++) NET_l[i] = 0.;
	for(i = 0; i<LROWS; i++) {
		for(j = 0; j<LCOLS; j++)
		{
			wx = w_l[i][j] * x[k][j];
			NET_l[i] = wx + NET_l[i];
		}
		z[i] = 1./(1.+exp(-N* NET_l[i]));
	}
	z[4] = -1;
	for(i = 0; i<KROWS; i++) NET_k[i] = 0.;
	for(i = 0; i<KROWS; i++)
	{
		for(j = 0; j<KCOLS; j++)
		{
			wz = w_k[i][j] * z[j];
			NET_k[i] = wz + NET_k[i];
		}
		OUT[i] = 1./(1. + exp(-N * NET_k[i]));
	}
}


void Backward_Pass()
{
	charge = 0.;
	for(i = 0; i<KROWS; i++)
	{
		charge = ((d[k][i] - OUT[j]) * (d[k][i] - OUT[i])) + charge;
	}
	E = ((1./2.) * charge) + E;
}

void Delta_Rule()
{
	for(i = 0; i<KROWS; i++)
	{
		delta_OUT[i] = (d[k][i] - OUT[i]) * (1. - OUT[i]) * OUT[i];
	}
	for(i = 0; i<KCOLS; i++)
	{
		delta_w = 0.;
		for(m = 0; m<KROWS; m++)
		{
			delta_w = (delta_OUT[m] * w_k[m][i]) + delta_w;
		}
		
		delta_z[i] = z[i] * (1. - z[i]) * delta_w;
	}
}

void Amend_Weight()
{
	for(i = 0; i<KROWS; i++)
	for(j = 0; j<KCOLS; j++) {
		w_k[i][j] = w_k[i][j] + (n*delta_OUT[i] * z[j]);
	}
	for(i = 0; i<LROWS; i++)
	for(j = 0; j<LCOLS; j++) 
	{
		w_l[i][j] = w_l[i][j] + (n * delta_z[i] * x[k][j]);
	}
}

void Input_x()
{
	do{
		printf("\nE_min = %f n = %f N = %f",EMIN,n,N);
		printf("\ntraining number = d , E = %f",1,E);
		Print_Weight();
		printf("\n\n");
		printf("--- X[0] = Carry In, x[1] = Input 1, x[2] = Input 2 ---\n");
		for(i = 0; i<3; i++)
		{
			printf("Input x[%d] of the Pattern X :",i);
			scanf("%f",&x[8][i]);
		}
		x[8][3] = -1.;
		Forward_Pass();
		printf("\n");
		for(i = 0; i<KROWS; i++)
		{
			printf("OUT[%d]= %f",i,OUT[i]);
		}
		printf("-- Where, OUT[0] = Carry Out, OUT[1]=Sum --");
		printf("\n\nanother Input ? [y/n]");
	}while((c = getch()) != 'n');
}


void Print_Weight()
{
	printf("\n");
	printf("\n[Amended Weight at l layer]");
	for(i =0 ; i<LROWS; i++)
	{
		printf("\n");
		for(j = 0; j<LCOLS; j++)
		{
			printf(" %+.3f",w_l[i][j]);
		}
	}
	printf("\n");
	printf("\n[Amended Weight at k layer]");
	for(i = 0; i<KROWS; i++)
	{
		printf("\n");
		for(j = 0; j<KCOLS; j++)
		{
			printf(" %+.3f",w_k[i][j]);
		}
	}
}

void Initialize_weight()
{
	int i,j;
	srand(7);
	for(i =0 ; i<LROWS; i++)
	for(j = 0; j<LCOLS; j++)
	{
		w_l[i][j] = ran();
	}
	
	for(i = 0; i<KROWS; i++)
	for(j = 0; j<KCOLS; j++) {
		w_k[i][j] = ran();
	}
}
