%%
load results/vae_capacity_data.mat

lambda = logspace(-3,-1,10);
filts = [4 8 16 32 48 64];
nsteps = 50;

data.loss_ce = data.loss_ce(2:end,:,:);
data.loss_nll = data.loss_nll(2:end,:,:);
filts = filts(2:end);

%%

leg = cell(1,length(filts));
for i = 1:length(filts)
  leg{i} = sprintf('%d',filts(i));
end

figure(1); clf;
imagesc(lambda,filts,mean(-data.loss_ce(:,:,end-nsteps+1:end),3));
xlabel('\lambda'); ylabel('filters/layer'); title('C'); colorbar;

figure(2); clf;
imagesc(lambda,filts,mean(-data.loss_nll(:,:,end-nsteps+1:end),3)./lambda);
xlabel('\lambda'); ylabel('filters/layer'); title('D'); colorbar;

%%

figure(3); clf;
cols = cbrewer('seq','Blues',length(filts)+2);
for i = 1:length(filts)
  semilogx(lambda, -mean(data.loss_ce(i,:,end-nsteps+1:end),3), ...
    'color',cols(i+2,:)); hold on;
end
semilogx(lambda, ones(size(lambda))*log2(3)*log(2), 'k--');
grid on; ylim([0.9 1.11]); set(gca,'fontsize',12);
xlabel('$$\lambda$$','interpreter','latex');
title('$$\mathcal{C}$$','interpreter','latex');
hleg = legend(leg,'location','sw'); set(gca,'fontname','Times New Roman');
title(hleg, 'Filters/layer'); set(gcf,'color','white');
export_fig figs/fig_vae_capacity_C.pdf

figure(4); clf;
cols = cbrewer('seq','Blues',length(filts)+2);
for i = 1:length(filts)
  semilogx(lambda, -mean(data.loss_nll(i,:,end-nsteps+1:end),3)./lambda, ...
    'color',cols(i+2,:)); hold on;
end
grid on; set(gca,'fontsize',12);
xlabel('$$\lambda$$','interpreter','latex');
title('$$\mathcal{D}$$','interpreter','latex');
hleg = legend(leg,'location','se'); set(gca,'fontname','Times New Roman');
title(hleg, 'Filters/layer'); set(gcf,'color','white');
export_fig figs/fig_vae_capacity_D.pdf