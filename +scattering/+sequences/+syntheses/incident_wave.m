%
% superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2019-04-06
% modified: 2020-04-02
%
classdef incident_wave

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        p_incident %( 1, 1 ) processing.field      % incident acoustic pressure field
        p_incident_grad ( 1, : ) processing.field	% spatial gradient of the incident acoustic pressure field

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = incident_wave( p_incident, p_incident_grad )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure class processing.field for p_incident

            % ensure nonempty p_incident_grad
%             if nargin < 2 || isempty( p_incident_grad )
%                 p_incident_grad = [];
%             end

            % ensure cell array for p_incident_grad
%             if ~iscell( p_incident_grad )
%                 p_incident_grad = { p_incident_grad };
%             end

            % ensure equal number of dimensions and sizes
%             auxiliary.mustBeEqualSize( p_incident, p_incident_grad );

            %--------------------------------------------------------------
            % 2.) create incident waves
            %--------------------------------------------------------------
            % repeat default incident wave
            objects = repmat( objects, size( p_incident ) );

            % iterate incident waves
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).p_incident = p_incident( index_object );
%                 objects( index_object ).p_incident_grad = p_incident_grad{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = incident_wave( p_incident, p_incident_grad )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

        %------------------------------------------------------------------
        % steered plane wave
        %------------------------------------------------------------------
        function incident_wave = compute_p_in_pw( incident_wave, setup, settings_tx_pw, v_d_unique )
% TODO: move to class sequence
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.syntheses.incident_wave
            

            %--------------------------------------------------------------
            % 2.) compute acoustic pressure
            %--------------------------------------------------------------
            % compute current complex-valued wavenumbers
            axis_k_tilde = compute_wavenumbers( setup.homogeneous_fluid.absorption_model, v_d_unique.axis );

            % compute incident acoustic pressure
            p_incident = double( v_d_unique.samples( :, settings_tx_pw.index_ref ) ) .* exp( -1j * axis_k_tilde.members * ( settings_tx_pw.e_theta.components * ( setup.FOV.shape.grid.positions - settings_tx_pw.position_ref )' ) );
            p_incident = physical_values.pascal( p_incident );

            % create field
            incident_wave.p_incident = processing.field( v_d_unique.axis, setup.FOV.shape.grid, p_incident );

        end % function incident_wave = compute_p_in_pw( incident_wave, setup, settings_tx_pw )

        %------------------------------------------------------------------
        % arbitrary wave
        %------------------------------------------------------------------
%         function field = compute_p_in
%         end
	end % methods (Access = private, Hidden)

end % classdef incident_wave
