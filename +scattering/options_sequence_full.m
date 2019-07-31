%
% superclass for all full sequence options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2019-07-30
%
classdef options_sequence_full < scattering.options_sequence

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_sequence_full( varargin )

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
            objects@scattering.options_sequence( size );

        end % function objects = options_sequence_full( varargin )

	end % methods

end % classdef options_sequence_full < scattering.options_sequence
