%
% superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2019-04-06
% modified: 2020-04-20
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

end % classdef incident_wave
