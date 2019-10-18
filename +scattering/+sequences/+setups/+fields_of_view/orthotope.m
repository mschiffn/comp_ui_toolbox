%
% superclass for all fields of view with orthotope shapes
%
% author: Martin F. Schiffner
% date: 2019-08-18
% modified: 2019-10-17
%
classdef orthotope < scattering.sequences.setups.fields_of_view.field_of_view

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthotope( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % class scattering.sequences.setups.geometry.orthotope ensures math.interval

            %--------------------------------------------------------------
            % 2.) create fields of view
            %--------------------------------------------------------------
            % create orthotopes
            orthotopes = scattering.sequences.setups.geometry.orthotope( varargin{ : } );

            % constructor of superclass
            objects@scattering.sequences.setups.fields_of_view.field_of_view( orthotopes );

        end % function objects = orthotope( varargin )

    end % methods

end % classdef orthotope < scattering.sequences.setups.fields_of_view.field_of_view
