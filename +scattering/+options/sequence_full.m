%
% superclass for all full sequence options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2019-08-03
%
classdef sequence_full < scattering.options.sequence

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence_full( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vectors for size
            % superclass ensures positive integers for size

            %--------------------------------------------------------------
            % 2.) create full sequence options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.sequence( size );

        end % function objects = sequence_full( varargin )

	end % methods

end % classdef sequence_full < scattering.options.sequence
