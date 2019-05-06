%
% superclass for all spatial discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-02-17
%
classdef spatial

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial( )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return empty object if no arguments
            if nargin == 0
                return;
            end

        end % function objects = spatial( )

        %------------------------------------------------------------------
        % compute hash values
        %------------------------------------------------------------------
        function str_hash = hash( spatials )

            % specify cell array for str_hash
            str_hash = cell( size( spatials ) );

            % iterate spatial discretizations
            for index_object = 1:numel( spatials )

                % use DataHash function to compute hash value
                str_hash{ index_object } = auxiliary.DataHash( spatials( index_object ) );

            end % for index_object = 1:numel( spatials )

            % avoid cell array for single spatial discretization
            if isscalar( spatials )
                str_hash = str_hash{ 1 };
            end

        end % function str_hash = hash( spatials )

	end % methods

end % classdef spatial
