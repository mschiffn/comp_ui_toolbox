%
% superclass for all memory sizes
%
% author: Martin F. Schiffner
% date: 2019-01-24
% modified: 2019-03-20
%
classdef memory < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = memory( bytes )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                bytes = 0;
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_value( bytes );

        end % function objects = memory( bytes )

        %------------------------------------------------------------------
        % convert to binary unit
        %------------------------------------------------------------------
        function size_binary = convert_binary( objects, base, exponent )

            % construct matrix of sizes
            size_binary = zeros( size( objects ) );

            % compute size in binary system
            for index_object = 1:numel( objects )
                size_binary( index_object ) = objects( index_object ).value / base^exponent;
            end

        end % function size_binary = convert_binary( objects, base, exponent )

        %------------------------------------------------------------------
        % convert to kilobyte (kB)
        %------------------------------------------------------------------
        function size_kilobyte = kilobyte( objects )

            % return result of binary conversion
            size_kilobyte = convert_binary( objects, 10, 3 );

        end % function size_kilobyte = kilobyte( objects )

        %------------------------------------------------------------------
        % convert to megabyte (MB)
        %------------------------------------------------------------------
        function size_megabyte = megabyte( objects )

            % return result of binary conversion
            size_megabyte = convert_binary( objects, 10, 6 );

        end % function size_megabyte = megabyte( objects )

        %------------------------------------------------------------------
        % convert to gigabyte (GB)
        %------------------------------------------------------------------
        function size_gigabyte = gigabyte( objects )

            % return result of binary conversion
            size_gigabyte = convert_binary( objects, 10, 9 );

        end % function size_gigabyte = gigabyte( objects )

        %------------------------------------------------------------------
        % convert to kibibyte (KiB)
        %------------------------------------------------------------------
        function size_kibibyte = kibibyte( objects )

            % return result of binary conversion
            size_kibibyte = convert_binary( objects, 2, 10 );

        end % function size_kibibyte = kibibyte( objects )

        %------------------------------------------------------------------
        % convert to mebibyte (MiB)
        %------------------------------------------------------------------
        function size_mebibyte = mebibyte( objects )

            % return result of binary conversion
            size_mebibyte = convert_binary( objects, 2, 20 );

        end % function size_mebibyte = mebibyte( objects )

        %------------------------------------------------------------------
        % convert to gibibyte (GiB)
        %------------------------------------------------------------------
        function size_gibibyte = gibibyte( objects )

            % return result of binary conversion
            size_gibibyte = convert_binary( objects, 2, 30 );

        end % function size_gibibyte = gibibyte( objects )

	end % methods

end % classdef memory < physical_values.physical_value
