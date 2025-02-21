clc, clear, close
load complexity.csv
M = complexity(:,1);
N = complexity(:,2);
error = complexity(:,3);
time_main = complexity(:,4);
time_grid = complexity(:,5);
expected_error = error(1)*(M(1)./M).^2;
G = 1.67
expected_comp = (M./M(1)).^G;
figure(1)
yyaxis left
loglog(M,error,'o','Color','blue')
hold on
grid on
loglog(M,expected_error,'Color','black')
xlabel('Nr of basis functions')
ylabel('Error')
hold off
yyaxis right
loglog(M,time_main/time_main(1),'o','Color','red')
hold on
loglog(M,expected_comp,'--','Color','black')
ylabel('complexity increase (w.r.t M=5)')
legend('Observed Error','1/n^2 decay','Observed complexity','n^{1.67} increase')